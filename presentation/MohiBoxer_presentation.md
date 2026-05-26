# MohiBoxer Präsentation

## 1. MohiBoxer

**Ein KI-Agent für Atari Boxing**

Kernaussage:

- Der Agent spielt selbstständig Atari Boxing.
- Er bekommt Spielbilder als Input.
- Er entscheidet daraus eine Aktion.
- Finale Evaluation: **99.2 Punkte Durchschnitt über 5 Spiele**.

Was ich sagen kann:

> Ich stelle heute MohiBoxer vor. Das ist mein KI-Agent für Atari Boxing. Der Agent sieht das Spiel als Bilder und entscheidet dann selbst, wie er sich bewegt oder schlägt. In der finalen Evaluation hat er im Durchschnitt 99.2 Punkte erreicht.

## 2. Ziel des Projekts

Das Ziel war einfach:

**Der Agent soll den eingebauten Atari-Gegner möglichst gut schlagen.**

Dafür braucht der Agent:

- Wahrnehmung: Wo bin ich? Wo ist der Gegner?
- Entscheidung: Welche Aktion ist jetzt gut?
- Training: Wie wird er besser?

Was ich sagen kann:

> Das Ziel war nicht einfach nur Code zu schreiben, sondern einen Agenten zu trainieren, der wirklich gut Boxing spielt. Er muss erkennen, wo der Gegner steht, und dann passend bewegen oder schlagen.

## 3. Was sieht der Agent?

Der Agent bekommt keine fertigen Informationen wie "Gegner links".

Er bekommt Bilder vom Spiel.

Diese Bilder werden vereinfacht:

- Farbe wird entfernt
- Bild wird auf `84x84` Pixel verkleinert
- 4 Bilder werden gestapelt

Warum 4 Bilder?

- Damit der Agent Bewegung erkennt.

Was ich sagen kann:

> Der Agent sieht nicht wie ein Mensch den Bildschirm in groß und farbig. Das Bild wird kleiner und grau gemacht. Außerdem bekommt er 4 Bilder hintereinander, damit er Bewegung erkennen kann.

## 4. Wie entscheidet der Agent?

Das Spiel hat **18 mögliche Aktionen**.

Beispiele:

- stehen bleiben
- bewegen
- schlagen
- bewegen und schlagen

Das Modell gibt 18 Werte aus.

Die Aktion mit dem höchsten Wert wird gewählt.

Was ich sagen kann:

> Das Modell bewertet alle 18 möglichen Aktionen. Danach nimmt der Agent einfach die Aktion mit dem höchsten Wert. So entscheidet er in jedem Spielschritt.

## 5. Erster Ansatz: DQN / DDQN

Der erste Plan war klassisches Reinforcement Learning:

- Agent probiert Aktionen aus.
- Er bekommt Rewards.
- Gute Aktionen sollen öfter gewählt werden.
- Replay Memory speichert Erfahrungen.

Problem:

**Das war zu langsam für Atari Boxing.**

Was ich sagen kann:

> Am Anfang war der Ansatz, dass der Agent alles durch Ausprobieren lernt. Das ist der klassische RL-Weg. Aber Boxing ist dafür schwer, weil man sehr viele Spiele braucht, bis gutes Verhalten entsteht.

## 6. Besserer Ansatz: Teacher-Student

Deshalb wurde ein Teacher-Student-Ansatz benutzt.

Idee:

- Ein starker Teacher-Agent kann schon gut spielen.
- Unser Modell ist der Student.
- Der Student lernt vom Teacher.

Einfach gesagt:

**Nicht komplett selbst herausfinden, sondern von einem guten Spieler lernen.**

Was ich sagen kann:

> Der wichtigste Wechsel war Teacher-Student Training. Das ist wie Nachhilfe: Ein guter Agent zeigt, welche Aktionen sinnvoll sind, und mein Modell lernt diese Entscheidungen nachzumachen.

## 7. Trainingsablauf

Der Ablauf war:

1. Teacher spielt Boxing.
2. Aktionen werden gespeichert.
3. Student lernt diese Aktionen.
4. Student spielt selbst.
5. Teacher korrigiert schwierige Situationen.
6. Das beste Modell wird gespeichert.

Was ich sagen kann:

> Der Student wurde nicht nur einmal trainiert. Er hat gespielt, wurde wieder korrigiert und weiter trainiert. Dadurch wurde er robuster, auch wenn er in schwierige Situationen kommt.

## 8. Neuronales Netz

Das Modell ist ein **CNN**.

Warum CNN?

- Der Input sind Bilder.
- CNNs sind gut für Bildverarbeitung.

Input:

- `4 x 84 x 84`

Output:

- `18` Werte, einer pro Aktion

Was ich sagen kann:

> Weil der Agent Bilder verarbeitet, ist ein CNN sinnvoll. Das Netz erkennt Muster in den Bildern, zum Beispiel Position und Abstand der Boxer. Am Ende kommen 18 Werte heraus.

## 9. Training in Zahlen

Finaler Trainingsumfang:

- **337** Trainingsrunden
- **20,202** gelabelte Episoden
- **8,966,962** Trainingsbeispiele

Wichtig:

- Nicht das neueste Modell wurde genommen.
- Das beste evaluierte Modell wurde genommen.

Was ich sagen kann:

> Nach jeder Trainingsphase wurde der Agent getestet. Für die Abgabe wurde nicht einfach das letzte Modell genommen, sondern das Modell mit der besten Evaluation.

## 10. Ergebnis

Finale Evaluation gegen den Standard-Gegner:

| Spiel | Reward |
| --- | ---: |
| 1 | 98 |
| 2 | 100 |
| 3 | 100 |
| 4 | 100 |
| 5 | 98 |

Durchschnitt:

**99.2**

Was ich sagen kann:

> Das finale Modell hat in 5 Spielen fast immer perfekt gespielt. Der Durchschnitt war 99.2 Punkte. Zusätzlich gibt es Videos von diesen Evaluationsspielen.

## 11. Was habe ich gelernt?

Gut funktioniert:

- Teacher-Student Training
- viele Trainingsdaten
- bestes Modell speichern

Schwierig:

- reines DDQN war zu langsam
- GPU-Setup war nicht einfach
- neuestes Modell war nicht automatisch bestes Modell

Was ich sagen kann:

> Die wichtigste Erkenntnis war, dass Teacher-Student Training für dieses Projekt viel effizienter war als reines Ausprobieren. Außerdem muss man Modelle sauber evaluieren, weil das letzte Modell nicht immer das beste ist.

## 12. Fazit

MohiBoxer:

- sieht Spielbilder
- nutzt ein CNN
- wählt aus 18 Aktionen
- wurde mit Teacher-Student Training verbessert
- erreicht **99.2 Punkte Durchschnitt**

Was ich sagen kann:

> Zusammengefasst ist MohiBoxer ein KI-Agent, der Atari Boxing über Spielbilder versteht und daraus Aktionen auswählt. Durch Teacher-Student Training konnte er sehr stark werden und gegen den Standard-Gegner fast perfekte Ergebnisse erreichen.
