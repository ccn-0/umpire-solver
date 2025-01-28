The traveling umpire problem solver
===================================

The Traveling Umpire Problem (TUP) considers the problem of assigning umpires (or referees) to a tournament of games. In a TUP instance, 2n teams are playing a double round robin tournament. Each team has a home venue, and every pair of teams plays twice in the tournament, once at each team's venue. There are 4n-2 time slots in the tournament; every team plays exactly once in each time slot. There are significant distances between home venues, defined by a symmetric distance matrix satisfying the triangle inequality.

There are n umpires to handle these games. Each game needs one umpire; each umpire handles one game per time slot. There are three main conflicting goals that drive the assignment of umpires to games. First, the umpires should not travel too much. Travel for the umpires is defined to be the sum of the distances between home venues of successive handled games. Second, umpires should see all the home venues. Third, umpires should not handle the games of a team too often in short succession.

The first goal is handled by the objective function: the objective is to minimize the total travel distance of the umpires. The second goal is handled as a hard constraint: every umpire must handle at least one game for each team at that team's home venue. The third goal is handled by two constraints with parameters to denote how closely to ideal the constraints must be met:

No umpire is in a home venue more than once in any n-d1 consecutive slots, and
No umpire sees a team (either home or away) more than once in any n/2-d2 (rounded down) consecutive slots.
The parameters d1 and d2 are tightness parameters: lower values of these parameters makes the corresponding constraint more difficult to satisfy.

For more information: https://benchmark.gent.cs.kuleuven.be/tup/en/

English Readme
==============

Install:

`pip install networkx`

Run:

`python tup.py [instance_file]`

Run example:

`python tup.py "instances/umps8.txt"`

The program prints average fitness and the best solution within the iteration limit

Czech Readme
============

Instalace:

Skript vyžaduje knihovnu networkx:

`pip install networkx`


Spouštění:

`python tup.py [instance_file]`


Příklad:

`python tup.py "instances/umps8.txt"`
