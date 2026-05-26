from __future__ import annotations
import argparse, json, os, shutil, traceback
from datetime import datetime, timezone
from pathlib import Path
BOXING_TF_DEVICE=os.environ.get("BOXING_TF_DEVICE","cpu").strip().lower()
if BOXING_TF_DEVICE=="cpu": os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import envpool
from boxing import BoxingConfig, evaluate
from distill_boxing import DistillationConfig, train_student
import expert.cleanrl_boxing_teacher as teacher_module
from expert.cleanrl_boxing_teacher import ATARI_MAX_FRAMES, DEFAULT_FILENAME, load_teacher

def now(): return datetime.now(timezone.utc).isoformat()
def append_event(control_dir: Path, event: str, payload: dict):
  control_dir.mkdir(parents=True, exist_ok=True)
  rec={"timestamp":now(),"event":event,"payload":payload}
  with (control_dir/"events.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")
  print(json.dumps(rec), flush=True)
def read_json(path: Path, default):
  if not path.exists(): return default
  with path.open("r",encoding="utf-8") as f: return json.load(f)
def write_json(path: Path, payload: dict):
  path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
  with tmp.open("w",encoding="utf-8") as f: json.dump(payload,f,indent=2)
  tmp.replace(path)
def sticky_env(seed:int):
  return envpool.make("Boxing-v5", env_type="gymnasium", num_envs=1, batch_size=1, episodic_life=True, repeat_action_probability=0.25, noop_max=30, full_action_space=False, max_episode_steps=ATARI_MAX_FRAMES, reward_clip=True, seed=seed)
def copy_if_exists(src: Path, dst: Path):
  if src.exists(): dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,dst)
def initialize_student_dir(student_dir: Path, initial_model: Path|None):
  student_dir.mkdir(parents=True, exist_ok=True)
  if (student_dir/"latest_student.h5").exists() or initial_model is None or not initial_model.exists(): return
  copy_if_exists(initial_model, student_dir/"latest_student.h5"); copy_if_exists(initial_model, student_dir/"initial_student.h5")
  copy_if_exists(initial_model.with_suffix(".keras"), student_dir/"latest_student.keras"); copy_if_exists(initial_model.with_suffix(".keras"), student_dir/"initial_student.keras")
def save_env_best(student_dir: Path, summary: dict):
  latest_h5=student_dir/"latest_student.h5"; latest_keras=student_dir/"latest_student.keras"
  best_h5=student_dir/"best_sticky_model.h5"; best_keras=student_dir/"best_sticky_model.keras"
  candidate_h5=student_dir/"friday_candidate.h5"; candidate_keras=student_dir/"friday_candidate.keras"
  for src,dst in [(latest_h5,best_h5),(latest_h5,candidate_h5),(latest_keras,best_keras),(latest_keras,candidate_keras)]: copy_if_exists(src,dst)
  payload=dict(summary); payload["best_sticky_model_h5"]=str(best_h5); payload["friday_candidate_h5"]=str(candidate_h5)
  write_json(student_dir/"best_sticky_evaluation.json", payload); return payload
def collect_student_labeled_trajectories(policy, student_model_path: Path, output_dir: Path, target_episodes: int, seed: int, shard_size: int = 2048) -> dict:
  import keras
  import numpy as np
  output_dir.mkdir(parents=True, exist_ok=True)
  existing_manifest=teacher_module._load_existing_manifest(output_dir)
  if existing_manifest is None and teacher_module._existing_shard_paths(output_dir): existing_manifest=teacher_module._recover_existing_progress(output_dir)
  starting_episode=int(existing_manifest.get("num_episodes",0)) if existing_manifest is not None else 0
  episodes_requested_this_run=int(target_episodes); target_total_episodes=starting_episode+episodes_requested_this_run
  episodic_returns=list(existing_manifest.get("episodic_returns",[])) if existing_manifest is not None else []
  env=sticky_env(seed=seed); model=keras.models.load_model(student_model_path, compile=False)
  observations_buffer=[]; actions_buffer=[]; rewards_buffer=[]; dones_buffer=[]; episode_ids_buffer=[]; logits_buffer=[]
  shard_index=teacher_module._next_shard_index(output_dir); current_return=0.0; current_episode=starting_episode
  try:
    observation,_=env.reset()
    while current_episode<target_total_episodes:
      observation=np.asarray(observation)
      teacher_logits=policy.predict_logits(observation)
      teacher_action=int(np.argmax(teacher_logits[0]))
      student_logits=model(observation, training=False).numpy()[0]
      student_action=int(np.argmax(student_logits))
      next_observation,reward,terminated,truncated,_=env.step(np.asarray([student_action], dtype=np.int32))
      done=bool(terminated[0]) or bool(truncated[0])
      observations_buffer.append(observation[0].copy()); actions_buffer.append(teacher_action); rewards_buffer.append(float(reward[0])); dones_buffer.append(done); episode_ids_buffer.append(current_episode); logits_buffer.append(teacher_logits[0].copy())
      current_return+=float(reward[0]); observation=next_observation
      if len(observations_buffer)>=shard_size:
        shard_index=teacher_module._flush_shard(output_dir,shard_index,observations_buffer,actions_buffer,rewards_buffer,dones_buffer,episode_ids_buffer,logits_buffer)
        observations_buffer.clear(); actions_buffer.clear(); rewards_buffer.clear(); dones_buffer.clear(); episode_ids_buffer.clear(); logits_buffer.clear()
      if done:
        episodic_returns.append(current_return); current_return=0.0; current_episode+=1
        teacher_module._write_manifest(output_dir, teacher_module._build_manifest(policy, output_dir, episodes_requested_this_run, target_total_episodes, episodic_returns, shard_size, True, seed, False, current_episode, shard_index))
        observation,_=env.reset()
  finally:
    env.close()
  shard_index=teacher_module._flush_shard(output_dir,shard_index,observations_buffer,actions_buffer,rewards_buffer,dones_buffer,episode_ids_buffer,logits_buffer)
  metadata=teacher_module._build_manifest(policy, output_dir, episodes_requested_this_run, target_total_episodes, episodic_returns, shard_size, True, seed, True, current_episode, shard_index)
  metadata["collection_mode"]="student_rollout_teacher_labeled"
  metadata["student_model_path"]=str(student_model_path)
  teacher_module._write_manifest(output_dir, metadata)
  return metadata

def parse_args():
  p=argparse.ArgumentParser()
  p.add_argument("--teacher-model",type=Path,default=Path("artifacts/teacher_cleanrl")/DEFAULT_FILENAME)
  p.add_argument("--dataset-dir",type=Path,default=Path("artifacts/teacher_boxing_dataset_sticky_rescue"))
  p.add_argument("--student-dir",type=Path,default=Path("artifacts/student_sticky_rescue"))
  p.add_argument("--evaluation-dir",type=Path,default=Path("artifacts/student_sticky_rescue_eval"))
  p.add_argument("--control-dir",type=Path,default=Path("artifacts/sticky_rescue_control"))
  p.add_argument("--initial-model",type=Path,default=Path("artifacts/student_imitation_server/latest_student.h5"))
  p.add_argument("--extra-teacher-model",type=Path,action="append",default=[])
  p.add_argument("--collect-episodes-per-round",type=int,default=200)
  p.add_argument("--train-epochs-per-round",type=int,default=3)
  p.add_argument("--dagger-episodes-per-round",type=int,default=20)
  p.add_argument("--evaluation-episodes",type=int,default=10)
  p.add_argument("--batch-size",type=int,default=256)
  p.add_argument("--learning-rate",type=float,default=1e-4)
  p.add_argument("--validation-fraction",type=float,default=0.05)
  p.add_argument("--shuffle-buffer",type=int,default=8192)
  p.add_argument("--seed",type=int,default=20260520)
  p.add_argument("--max-rounds",type=int,default=0)
  return p.parse_args()
def main():
  args=parse_args(); stop_file=args.control_dir/"STOP_STICKY_RESCUE"; state_path=args.control_dir/"run_state.json"
  args.control_dir.mkdir(parents=True,exist_ok=True); args.dataset_dir.mkdir(parents=True,exist_ok=True); initialize_student_dir(args.student_dir,args.initial_model)
  state=read_json(state_path,{"status":"starting","round_index":0,"best_sticky_reward":-9999.0,"best_model_path":None,"updated_at":now()})
  state.update({"status":"running","updated_at":now()}); write_json(state_path,state)
  append_event(args.control_dir,"rescue_started",{"teacher_model":str(args.teacher_model),"dataset_dir":str(args.dataset_dir),"student_dir":str(args.student_dir),"evaluation_dir":str(args.evaluation_dir),"control_dir":str(args.control_dir),"initial_model":str(args.initial_model),"collect_episodes_per_round":args.collect_episodes_per_round,"train_epochs_per_round":args.train_epochs_per_round,"evaluation_episodes":args.evaluation_episodes,"batch_size":args.batch_size,"learning_rate":args.learning_rate})
  teacher_module.build_env=sticky_env
  teacher_paths=[args.teacher_model]+list(args.extra_teacher_model)
  teachers=[(path, load_teacher(path,seed=args.seed+i)) for i,path in enumerate(teacher_paths)]
  append_event(args.control_dir,"teacher_pool_loaded",{"teacher_models":[str(path) for path,_ in teachers]})
  rounds_run=0
  while not stop_file.exists():
    if args.max_rounds and rounds_run>=args.max_rounds: break
    round_index=int(state.get("round_index",0))+1; rounds_run+=1
    state.update({"round_index":round_index,"active_phase":"collect_sticky_teacher","updated_at":now()}); write_json(state_path,state); append_event(args.control_dir,"collect_started",{"round_index":round_index})
    collection_summaries=[]
    base=args.collect_episodes_per_round//max(1,len(teachers)); remainder=args.collect_episodes_per_round%max(1,len(teachers))
    for teacher_i,(teacher_path,teacher_policy) in enumerate(teachers):
      episodes_to_collect=base+(1 if teacher_i<remainder else 0)
      if episodes_to_collect<=0: continue
      append_event(args.control_dir,"collect_teacher_started",{"round_index":round_index,"teacher_model":str(teacher_path),"episodes":episodes_to_collect})
      summary=teacher_module.collect_trajectories(policy=teacher_policy, output_dir=args.dataset_dir, target_episodes=episodes_to_collect, seed=args.seed+round_index*1000+teacher_i*100000, shard_size=2048, append=True)
      summary["source_teacher_model"]=str(teacher_path)
      collection_summaries.append(summary)
      append_event(args.control_dir,"collect_teacher_completed",{"round_index":round_index,"teacher_model":str(teacher_path),"summary":summary})
    if args.dagger_episodes_per_round>0 and (args.student_dir/"latest_student.h5").exists():
      dagger_teacher_path,dagger_teacher=teachers[(round_index-1)%len(teachers)]
      append_event(args.control_dir,"dagger_collect_started",{"round_index":round_index,"teacher_model":str(dagger_teacher_path),"episodes":args.dagger_episodes_per_round})
      dagger_summary=collect_student_labeled_trajectories(dagger_teacher, args.student_dir/"latest_student.h5", args.dataset_dir, args.dagger_episodes_per_round, args.seed+round_index*5000)
      dagger_summary["source_teacher_model"]=str(dagger_teacher_path)
      collection_summaries.append(dagger_summary)
      append_event(args.control_dir,"dagger_collect_completed",{"round_index":round_index,"summary":dagger_summary})
    collection_summary=collection_summaries[-1] if collection_summaries else {}
    state.update({"total_sticky_teacher_episodes":collection_summary.get("num_episodes"),"total_sticky_teacher_samples":collection_summary.get("num_samples"),"active_phase":"train_student","updated_at":now()}); write_json(state_path,state); append_event(args.control_dir,"collect_completed",{"round_index":round_index,"summaries":collection_summaries})
    train_summary=train_student(DistillationConfig(dataset_dir=args.dataset_dir, output_dir=args.student_dir, epochs=args.train_epochs_per_round, batch_size=args.batch_size, learning_rate=args.learning_rate, validation_fraction=args.validation_fraction, shuffle_buffer=args.shuffle_buffer, seed=args.seed+round_index, resume_model=args.student_dir/"latest_student.h5", round_index=round_index))
    state.update({"active_phase":"evaluate_sticky_student","updated_at":now()}); write_json(state_path,state); append_event(args.control_dir,"train_completed",{"round_index":round_index,"summary":train_summary})
    eval_config=BoxingConfig(eval_episodes=args.evaluation_episodes, seed=args.seed+round_index*17, full_action_space=True, repeat_action_probability=0.25)
    evaluation_summary=evaluate(config=eval_config, output_dir=args.evaluation_dir/f"round_{round_index:05d}", algorithm="dqn", model_path=args.student_dir/"latest_student.h5", checkpoint_path=None, episodes=args.evaluation_episodes, render_mode=None, record_video=False)
    avg=float(evaluation_summary.get("average_reward",-9999.0)); improved=avg>float(state.get("best_sticky_reward",-9999.0))
    if improved:
      best_payload=save_env_best(args.student_dir,evaluation_summary); state["best_sticky_reward"]=avg; state["best_model_path"]=best_payload["friday_candidate_h5"]; state["best_summary_path"]=str(args.student_dir/"best_sticky_evaluation.json")
    state.update({"last_completed_round":round_index,"last_eval_reward":avg,"last_eval_summary":evaluation_summary,"active_phase":"idle_between_rounds","updated_at":now()}); write_json(state_path,state)
    append_event(args.control_dir,"eval_completed",{"round_index":round_index,"improved":improved,"average_reward":avg,"best_sticky_reward":state.get("best_sticky_reward"),"summary":evaluation_summary})
  state["status"]="stopped" if stop_file.exists() else "completed"; state["active_phase"]=None; state["updated_at"]=now(); write_json(state_path,state); append_event(args.control_dir,"rescue_stopped",state)
if __name__=="__main__":
  try: main()
  except Exception as exc:
    append_event(Path("artifacts/sticky_rescue_control"),"fatal_error",{"error":repr(exc),"traceback":traceback.format_exc()}); raise
