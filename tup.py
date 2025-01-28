import re
import random
from operator import itemgetter
import sys

import networkx as nx
from networkx.algorithms.bipartite.matching import minimum_weight_full_matching
from networkx.algorithms.bipartite.matching import maximum_matching

MutationProbability = 0.05
InitPopulation = 500
GenerationCount = 1000000

# Constraint (4) parameter
constraintHomeParam = 0 

# Constraint (5) parameter
constraintSeenParam = 0 

# Experimentally chosen penalty factor for constraint violations.
PENALTY = 1000000


def parseInstance(filename):
    with open(filename) as f:
        raw = f.read()

    numberMatch = re.findall(r'(-{0,1}\d+)', raw)

    n = int(numberMatch[0])
    dist = list(map(int, numberMatch[1:n**2 + 1]))
    opps = list(map(int, numberMatch[n**2 + 1:]))

    return n, dist, opps


def getGamePairs(opps):

    slots = []

    for s in range(2*teamCount - 2):
        currentSlot = []
        for m in range(teamCount):
            cur = opps[s*teamCount + m]
            # only get home matches
            if cur > 0:
                currentSlot.append( (m+1,cur) )
        slots.append(currentSlot)

    return slots

# Scores a full solution
# Returns a pair of score sum and list of individual umpire scores
def fitness(solution):

    schedules = [[slot[u] for slot in solution] for u in range(umpCount)]

    # Score each umpires schedule
    scores = [scoreUmpire(s) for s in schedules]

    return sum(scores), scores


def scoreUmpire(schedule):

    currentHome, currentAway = schedule[0]
    score = 0

    # Set of visited homes for constraint (3) validation
    visitedHomes = set()
    visitedHomes.add(currentHome)

    # Constraint (4)
    # Saves last visited homes up to 'constraintHome' slots
    homeHistory = [currentHome]

    # Constraint (5)
    # Saves last seen teams up to 'constraintSeen' slots
    seenHistory = [(currentHome, currentAway)]

    # Go through the schedule and count score
    for home, away in schedule[1:]:
        # Add distance travel
        i = (currentHome - 1) + (home - 1) * teamCount
        score += distanceTable[i]
        # Constraint (3) - adding a visited home
        visitedHomes.add(home)

        # Constraint (4) check
        homeHistory.append(home)
        homeHistory = homeHistory[-constraintHome:]
        if len(set(homeHistory)) != len(homeHistory):
            # Set of past seen homes is not unique, apply penalty
            score += PENALTY

        # Constraint (5) check
        seenHistory.append((home, away))
        seenHistory = seenHistory[-constraintSeen:]
        seenHistoryFlatten = list(sum(seenHistory, ()))
        if len(set(seenHistoryFlatten)) != len(seenHistoryFlatten):
            # Set of past seen teams is not unique, apply penalty
            score += PENALTY

        currentHome = home

    # Constraint (3) check, did this umpire visit all homes?
    if len(visitedHomes) < teamCount:
        # Penalty for every home not visited
        score += PENALTY * (teamCount - len(visitedHomes))

    return score


# Get score of a umpire's schedule, which is given as left and right schedule
# segment. Applies big penalties for constraint violations.
def scoreUmpireSeg(left, right):

    # Get distance traveled on crossover slot as initial score
    leftHome, _ = left[-1]
    rightHome, _ = right[0]
    i = (leftHome - 1) + (rightHome - 1) * teamCount
    score = distanceTable[i]

    # Join both segments to get the whole schedule
    schedule = left + right
    currentHome, currentAway = schedule[0]

    # Set of visited homes for constraint (3) validation
    visitedHomes = set()
    visitedHomes.add(currentHome)

    # Constraint (4)
    # Saves last visited homes up to 'constraintHome' slots
    homeHistory = [currentHome]

    # Constraint (5)
    # Saves last seen teams up to 'constraintSeen' slots
    seenHistory = [(currentHome, currentAway)]

    # Go through the schedule and search for constraint violations
    for home, away in schedule[1:]:
        # Constraint (3) - adding a visited home
        visitedHomes.add(home)

        # Constraint (4) check
        homeHistory.append(home)
        homeHistory = homeHistory[-constraintHome:]
        if len(set(homeHistory)) != len(homeHistory):
            # Set of past seen homes is not unique, apply penalty
            score += PENALTY

        # Constraint (5) check
        seenHistory.append((home, away))
        seenHistory = seenHistory[-constraintSeen:]
        seenHistoryFlatten = list(sum(seenHistory, ()))
        if len(set(seenHistoryFlatten)) != len(seenHistoryFlatten):
            # Set of past seen teams is not unique, apply penalty
            score += PENALTY

        currentHome = home

    # Constraint (3) check, did this umpire visit all homes?
    if len(visitedHomes) < teamCount:
        # Penalty for every home not visited
        score += PENALTY * (teamCount - len(visitedHomes))

    return score, schedule


# Locally-optimized crossover. Pick a random slot where the crossover
# happens, then solve a graph matching problem that minimizes score.
# Produces one new solution.
def crossover(s1, s2, rs=None):

    # Pick a random slot (first slot excluded)
    if rs == None:
        rs = random.randrange(1, slotCount)

    # Parent s1 becomes the left segments and parent s2 become right segments.
    leftSegments = s1[:rs]
    rightSegments = s2[rs:]

    # Make a bipartite graph
    B = nx.Graph()
    B.add_nodes_from(list(range(umpCount)), bipartite=0)
    B.add_nodes_from(list(range(umpCount, umpCount*2)), bipartite=1)

    # List of all possible schedules per umpire
    schedules = [[] for _ in range(umpCount)]

    # For every umpire make all possible schedules and score them.
    for u in range(umpCount):
        for s in range(umpCount):
            left = [slot[u] for slot in leftSegments]
            right = [slot[s] for slot in rightSegments]
            score, schedule = scoreUmpireSeg(left, right)
            schedules[u].append(schedule)
            B.add_edge(u, umpCount + s, weight = score)

            # print(f"ump {u}, seg {s}: {score}")
            # print(f"sched: {left + right}")

    # Solve minimum matching bipartite graph problem
    optimal_matching = minimum_weight_full_matching(B)

    optimal_schedules = []
    # Pick the optimized schedules
    for u in range(umpCount):
        i = optimal_matching[u] - umpCount
        optimal_schedules.append(schedules[u][i])

    solution = [[sched[s] for sched in optimal_schedules] 
                for s in range(slotCount)
               ]

    return solution 


# We randomly select two games in a randomly selected slot
# and flip the umpires assigned to these games. The mutation operator 
# introduces a certain amount of randomness to the search. It
# can help identify solutions that crossover alone might not.
def mutate(solution):

    # Roll if the solution gets mutated
    if random.random() > MutationProbability:
        return solution

    # Pick a random slot
    sn = len(solution)
    rs = random.randrange(sn)

    # Swap 2 random games in that slot
    a, b = random.sample(list(range(len(solution[rs]))), 2)
    solution[rs][a], solution[rs][b] = solution[rs][b], solution[rs][a]

    # print(f" > mutate picked slot {rs} and swapped {a} {b}")

    return solution

# Build an initial population of solutions. These are built using a
# randomized greedy search algorithm. The quality of the solutions should
# be sufficient for use in the implemented GA.
def generateInitPopulation(initSolution):

    population = []
    hashes = []

    # Create an initial population 
    for n in range(InitPopulation):
        solution = pgm(initSolution[:1], initSolution[1:])
        population.append(solution)
        h = hashSolution(solution)
        hashes.append(h)

    return population, hashes


# Solve perfect matching for left segments and next slot segments.
# Choose either best matching or matching with penalties.
# TODO
def pgm(leftSegments, rightSegments):

    # Schedule finished
    if rightSegments == []:
        return leftSegments

    nextSlot = rightSegments[0]

    # Make a bipartite graph
    B = nx.Graph()
    B.add_nodes_from(list(range(umpCount)), bipartite=0)
    B.add_nodes_from(list(range(umpCount, umpCount*2)), bipartite=1)

    # List of all possible partial schedules per umpire
    partialSchedules = [[] for _ in range(umpCount)]

    # For every umpire make all possible schedules and score them.
    # Pick only those that don't violate (4) and (5)
    for u in range(umpCount):
        for s in range(umpCount):
            left = [slot[u] for slot in leftSegments]
            game = nextSlot[s]
            score, partialSchedule = scorePartialSchedule(left, game)
            partialSchedules[u].append(partialSchedule)
            # Assignment doesn't violate any constraints
            if score != -1:
                # Experimental: scale the score randomly so it isn't always optimal
                B.add_edge(u, umpCount + s, weight = int(score * random.uniform(1.0, 1.2)))
            # Some violation happened, make a high random weight
            else:
                B.add_edge(u, umpCount + s, weight = PENALTY * random.randint(1,5))

    matching = minimum_weight_full_matching(B)

    # Build new partial schedules based on the previous matching
    newSchedules = []
    for u in range(umpCount):
        i = matching[u] - umpCount
        newSchedules.append(partialSchedules[u][i])

    newLeftSegments = [[sched[s] for sched in newSchedules] 
                for s in range(len(newSchedules[0]))
               ]

    # Recursively run for next slot
    solution = pgm(newLeftSegments, rightSegments[1:])

    return solution



# Left is a partial schedule of an umpire
# Game is a single game that will be appended to this partial schedule.
def scorePartialSchedule(left, game):

    # Get distance traveled from last slot in partial schedule and next game.
    leftHome, _ = left[-1]
    rightHome, _ = game
    i = (leftHome - 1) + (rightHome - 1) * teamCount
    score = distanceTable[i]

    # Join the next game to the partial schedule
    partialSchedule = left + [game]
    currentHome, currentAway = partialSchedule[0]

    # Constraint (4)
    # Saves last visited homes up to 'constraintHome' slots
    homeHistory = [currentHome]

    # Constraint (5)
    # Saves last seen teams up to 'constraintSeen' slots
    seenHistory = [(currentHome, currentAway)]

    # Go through the partial schedule and search for constraint violations.
    # If a violation is found, exit with score -1 that signalizes that this
    # matching is impossible.
    for home, away in partialSchedule[1:]:

        # Constraint (4) check
        homeHistory.append(home)
        homeHistory = homeHistory[-constraintHome:]
        if len(set(homeHistory)) != len(homeHistory):
            # Set of past seen homes is not unique
            return -1, partialSchedule

        # Constraint (5) check
        seenHistory.append((home, away))
        seenHistory = seenHistory[-constraintSeen:]
        seenHistoryFlatten = list(sum(seenHistory, ()))
        if len(set(seenHistoryFlatten)) != len(seenHistoryFlatten):
            # Set of past seen teams is not unique
            return -1, partialSchedule

        currentHome = home

    # No constraint was violated, return score as distance traveled.
    return score, partialSchedule


# Get a hash of a solution, used for quick lookup
def hashSolution(solution):
    h = hash(str(solution))
    return h


def main():

    n, dist, opps = parseInstance(sys.argv[1])

    global teamCount
    global umpCount
    global slotCount
    global distanceTable

    global constraintHome
    global constraintSeen

    teamCount = n
    umpCount = n//2
    slotCount = 2*n - 2
    distanceTable = dist

    # Constraint (4)
    constraintHomeBase = umpCount  
    constraintHome = constraintHomeBase - constraintHomeParam
    if constraintHome < 1:
        constraintHome = 1

    # Constraint (5)
    constraintSeenBase = umpCount//2
    constraintSeen = constraintSeenBase - constraintSeenParam
    if constraintSeen < 1:
        constraintSeen = 1

    # Read input opponents matrix and make an initial solution, which will be
    # used for population builder
    initSolution = getGamePairs(opps)


    # Run
    bestEver = 9999999999
    bestEverGenN = 0
    population, hashes = generateInitPopulation(initSolution)
    for generation in range(GenerationCount):

        n = 0 # Count new solutions in this generation
        while n < (InitPopulation//2):
            
            # Randomly pick 2 different solutions and do crossover
            s1, s2 = random.sample(population, 2)
            newSolution = crossover(s1, s2)
            newh = hashSolution(newSolution)
            # Check if this new solution is unique in this population and add it
            if newh not in hashes:
                population.append(newSolution)
                hashes.append(newh)
            n += 1

        # Get fitness of all solutions
        scoredSolutions = []
        for i, solution in enumerate(population):
            scoredSolutions.append(
                (fitness(solution)[0], solution)
                )

        # Pick best 'InitPopulation' number of solutions
        scoredSolutions = sorted(scoredSolutions, key=itemgetter(0))[:InitPopulation]
        averageFitness = sum([s[0] for s in scoredSolutions]) / InitPopulation

        if scoredSolutions[0][0] < bestEver:
            bestEver = scoredSolutions[0][0]
            bestEverGenN = generation
            print(f"gen {generation} found new best: {bestEver}")
            print(scoredSolutions[0][1])
        print(f"gen {generation}, average fitness: {averageFitness}, best so far: {bestEver}")


        population = [s[1] for s in scoredSolutions]

        # Apply mutations
        population = [mutate(s) for s in population]
        # New hashes
        hashes = [hashSolution(s) for s in population]



if __name__ == "__main__":
    main()