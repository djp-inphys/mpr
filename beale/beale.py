import numpy as np
import itertools

def beale(a, N):
    # a: the correlation Matrix
    # N: number of variable to keep
    # EPLISON = 1e-12
    numVar = len(a)-1

    uncondThresholds = CalUnconditionalThresholds(a, numVar)

    MLEVEL  = -1 * np.ones(numVar)
    MBEST   = -1 * np.ones(numVar)
    NOIN    = 0
    BRSSQ   = float('inf')
    MAXOD   = 1

    subSetSelCompleted = False
    while (not subSetSelCompleted):
        # %Scan the list of variables which are out of
        # %the equation with MLEVEL = — 1 to see
        # %which would give the largest reduction
        # %in RSSQ when introduced
        selectedVar = SelectVariable(a, MLEVEL)
        # Found a variable ?
        if selectedVar != -1:
            if NOIN == N - 1:
                # %If the chosen variable were introduced
                # %into the equation, would the
                # %RSSQ be less than BRSSQ ?
                # % this time the selected variable is not real pivoted,
                # % it is just a trial, therefore the operation is upon a
                # % temperay matrix temp, 'matrix a' stays the same
                temp = pivotIn(a, selectedVar)
                if temp[numVar, numVar] < BRSSQ:
                    # %Update BRSSQ
                    # %Update MBEST by setting:
                    # %MBEST = 1 for chosen variable,
                    # %MBEST = MLEVEL otherwise
                    BRSSQ = temp[numVar, numVar]
                    for i in range(numVar):
                        if i == selectedVar:
                            MBEST[i] = 1
                        else:
                            MBEST[i] = MLEVEL[i]

                    # %Examine the unconditional thresholds
                    # %of all variables with
                    # %MLEVEL > 0. If any such variable
                    # %has an unconditional
                    # %threshold 2s BRSSQ, set the
                    # %corresponding MLEVEL = 0
                    for i in range(numVar):
                        if MLEVEL[i] > 0 and uncondThresholds[i] > BRSSQ:
                            MLEVEL[i] = 0
            elif NOIN < N - 1:  # %if NOIN == N-1:
                # %Pivot the chosen variable into the equation,
                # %setting the corresponding MLEVEL = MAX0D + 2.
                # %N0LN = N0IN+1
                a = pivotIn(a, selectedVar)
                MLEVEL[selectedVar] = MAXOD + 2
                NOIN = NOIN + 1
                continue

        MAXOD = CalMaxOdd(MLEVEL)

        if MAXOD >= 3:
            # %Pivot out all those variables with MLEVEL > MAX0D, setting them
            # %out at MLEVEL = — 1. Also set out at MLEVEL = - 1 all those
            # %variables with MLEVEL =S -MAX0D. Update N0LN
            for i in range(numVar):
                if MLEVEL[i] > MAXOD:
                    MLEVEL[i] = -1
                    a = pivotOut(a, i)
                    NOIN = NOIN - 1

                if MLEVEL[i] <= -MAXOD:
                    MLEVEL[i] = -1
                    # %Pivot out that variable which has MLEVEL = MAX0D and which
                    # %gives the minimum increase in the RSSQ when removed. Set the
                    # %corresponding MLEVEL = -MAX0D.N0IN = N0IN- 1

            minimunIncrease = float('inf')
            minRssqIncreaseVarIndex = -1
            for i in range(numVar):
                if MLEVEL[i] == MAXOD:
                    increase = np.abs( (a[numVar, i]**2)/a[numVar, numVar] )
                    if increase < minimunIncrease:
                        minimunIncrease = increase
                        minRssqIncreaseVarIndex = i

            a = pivotOut(a, minRssqIncreaseVarIndex)
            MLEVEL[minRssqIncreaseVarIndex] = -MAXOD
            NOIN = NOIN - 1
            # % Examine the conditional thresholds of all variables with
            # %MLEVEL = — 1. If any such variable has a conditional threshold
            # %> BRSSQ pivot the variable into the equation and set the corresponding
            # %MLEVEL = MAX0D + 1. Update N0IN
            condThresholds = CalcConditionalThresholds(a, MLEVEL)

            for i in range(numVar):
                if MLEVEL[i] == -1 and condThresholds[i] >= BRSSQ and NOIN < N - 1:
                    a = pivotIn(a, i)
                    MLEVEL[i] = MAXOD + 1
                    NOIN = NOIN + 1
        else:
            subSetSelCompleted = True

    selected = np.zeros(numVar)
    for i in range(numVar):
        if MBEST[i] >= 0:
            selected[i] = 1

    return selected



def CalUnconditionalThresholds(a, numVar):
    EPLISON = 1e-12
    varpivoted = np.zeros(numVar)
    uncondThresholds = np.zeros(numVar)

    acopy = np.copy(a)
    for q in range(numVar):
        if acopy[q, q] > EPLISON:
            acopy = pivotIn(acopy, q)
            varpivoted[q] = 1

    for q in range(numVar):
        if varpivoted[q] > 0:
            temp = pivotOut(acopy, q)
            uncondThresholds[q] = temp[numVar, numVar]
        else:
            uncondThresholds[q] = 0

    return uncondThresholds


def pivotOut(a, q):
    numVar = len(a)
    b = np.zeros((numVar, numVar))

    b[q, q] = -1 / a[q, q]

    for j in range(numVar):
        if j != q:
            b[j, q] = - a[j, q] * b[q, q]
            b[q, j] = b[j, q]

    for j in range(numVar):
        for k in range(numVar):
            if j != q and k != q:
                b[j, k] = a[j, k] - (a[j, q] * b[q, k])
                b[k, j] = b[j, k]

    return b

def pivotIn(a, q):
    try:
        numVar  = len(a)
        b       = np.zeros((numVar, numVar))

        b[q, q] = -1 / a[q, q]

        for j in range(numVar):
            if j != q:
                b[j, q] = a[j, q] * b[q, q]
                b[q, j] = b[j, q]

        for j in range(numVar):
            for k in range(j, numVar):
                if j != q and k != q:
                    b[j, k] = a[j, k] + (a[j, q] * b[q, k])
                    b[k, j] = b[j, k]
    except:
        print( "Error" )


    return b




def SelectVariable(a, MLEVEL):
    EPSILON     = 1e-12
    selectedVar = -1
    mostRssDec  = 0
    numVar      = len(MLEVEL)
    acopy       = np.copy(a)

    for i in range(numVar):
        if MLEVEL[i] == -1 and acopy[i, i] > EPSILON:
            residualChange = np.abs( (acopy[numVar, i]**2)/acopy[i, i] )
            if residualChange > mostRssDec:
                mostRssDec = residualChange
                selectedVar = i

    return selectedVar


def CalMaxOdd(MLEVEL):
    MAXOD   = -1
    numVar  = len(MLEVEL)

    for i in range(numVar):
        if MLEVEL[i] % 2 and MLEVEL[i] > MAXOD:
            MAXOD = MLEVEL[i]

    return MAXOD


#function
def CalcConditionalThresholds(a, MLEVEL):
    EPSILON = 1e-12
    acopy  = np.copy(a)
    numVar = len(MLEVEL)
    pivoted = np.zeros(numVar)

    for i in range( numVar ):
        if (MLEVEL[i] == -1) and (acopy[i, i] > EPSILON):
            acopy       = pivotIn(acopy, i)
            pivoted[i]  = 1

    condThresholds = np.zeros(numVar)

    for i in range( numVar ):
        if (pivoted[i] == 1):
            temp = pivotOut(acopy, i)
            condThresholds[i] = temp[numVar, numVar]

    return condThresholds


def exhaustive (a, N):
    numVar  = len(a) - 1
    indices =  list( itertools.combinations(list( range(numVar) ),N) )

    BRSSQ   = float('inf')

    for tup in indices:
        acopy = np.copy(a)
        for vrbl in tup:
            acopy = pivotIn(acopy, vrbl)

        if(acopy[numVar,numVar]<BRSSQ):
            bestIndices = tup
            BRSSQ = acopy[numVar,numVar]


    selected = np.zeros(numVar, dtype='int' )
    selected[list(bestIndices)] = 1

    return selected.tolist()