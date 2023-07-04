import time
import numpy as np
import math

def printEnd():
    endtime = time.perf_counter()
    interval = (endtime - starttime) * 1000
    print(f'{interval:.3f} [mec]')

def CalcL_Func(dictSignal, dictCommonParam, dictDetectorParam, LaserParam, Spectrum):
    DofRow = dictSignal['Dof']
    Dm = Spectrum['aryDm']
    Dz = Spectrum['aryDz']
    numN = dictCommonParam['numN']
    kd = dictDetectorParam['kd']
    D0 = LaserParam['D0']
    Rz = LaserParam['aryRz']
    Rs = LaserParam['aryRs']

    Dof = np.average(DofRow)

    # 直線性補正
    Dm = (Dm - D0 - Dof) + kd * np.square(Dm - D0 - Dof)
    Dz = (Dz - D0 - Dof) + kd * np.square(Dz - D0 - Dof)

    # 特徴成分(ゼロ) - 演算
    Lz = (-1) * np.log(Dz)
    Fz = np.zeros(numN)
    for f in range(0, numN):
        Fz[f] = np.sum(Lz * Rs[:, f] - Lz * Rz)

    # 特徴成分(スパン) - 演算
    Lm = (-1) * np.log(Dm)
    Fs = np.zeros(numN)
    for f in range(0, numN):
        Fs[f] += np.sum(Lm * Rs[:, f] - Lm * Rz)

    # 特徴成分 - 演算
    F = np.zeros(numN)
    for f in range(0, numN):
        F[f] = (Fs[f] - Fz[f]) / 500

    return F

def CalcSts_Func(dictLaserParam, Ts):
    lowTs = dictLaserParam['maxTs'] * (-1)
    highTs = dictLaserParam['maxTs']
    numTableTs = dictLaserParam['numTableTs']
    S = dictLaserParam['aryParamS']

    if Ts <= lowTs:
        Sts = S[0]
    elif Ts >= highTs:
        Sts = S[numTableTs - 1]
    else:
        intTs = int((Ts - lowTs) / WIDTH_TS)
        decTs = (Ts / WIDTH_TS) - math.floor((Ts / WIDTH_TS))
        lowTsS = S[intTs]
        highTsS = S[intTs + 1]
        Sts = lowTsS + ((highTsS - lowTsS) * decTs)

    return Sts

def CalcS_Func(dictSignal, dictCommonParam, dictLaserParam, Bf, Ts):
    P = dictSignal['P']
    lowP = dictLaserParam['lowP']
    highP = dictLaserParam['highP']
    numTableP = dictLaserParam['numTableP']
    numN = dictCommonParam['numN']

    Sts = CalcSts_Func(dictLaserParam, Ts)
    aryP = P * Bf

    i = 0
    Sb = np.c_[np.zeros(numN)]
    for P in aryP:
        if P <= lowP:
            colSp = Sts[0][:, i]
        elif P >= highP:
            colSp = Sts[numTableP - 1][:, i]
        else:
            intP = int(P - lowP)
            decP = P - int(P)
            lowPressS = Sts[intP][:, i]
            highPressS = Sts[intP + 1][:, i]
            colSp = lowPressS + ((highPressS - lowPressS) * decP)


        colSb = np.c_[colSp / Bf[i]]
        Sb = np.hstack([Sb, colSb])
        i = i + 1

    Sb = np.delete(Sb, 0, axis=1)

    return Sb

def CLS_Func(dictSignal, dictCommonParam, dictLaserParam, L):
    numNc = dictCommonParam['numNc']
    deltaBf = dictLaserParam['deltaBf']
    deltaTs = dictLaserParam['deltaTs']
    minBf = dictLaserParam['minBf']
    maxBf = dictLaserParam['maxBf']
    maxTs = dictLaserParam['maxTs']

    Ts = 0.00
    Bf = np.full([MAX_COMP], 1.0)

    Sc = CalcS_Func(dictSignal, dictCommonParam, dictLaserParam, Bf, Ts)
    if dictCommonParam['numComp'] == 3:
        Sc[:, 3] = 0
    elif dictCommonParam['numComp'] == 2:
        Sc[:, 3] = 0
        Sc[:, 2] = 0
    elif dictCommonParam['numComp'] == 1:
        Sc[:, 3] = 0
        Sc[:, 2] = 0
        Sc[:, 1] = 0

    pinvS = np.linalg.pinv(Sc)
    # pinvS = Sc.T
    C = pinvS.dot(L)

    for loop in range(0, numNc):
        Sbt = CalcS_Func(dictSignal, dictCommonParam, dictLaserParam, Bf, Ts)
        SbDelta = CalcS_Func(dictSignal, dictCommonParam, dictLaserParam, Bf + deltaBf, Ts)
        StDelta = CalcS_Func(dictSignal, dictCommonParam, dictLaserParam, Bf, Ts + deltaTs)

        SbAdd = ((SbDelta - Sbt) / deltaBf)
        for k in range(0, MAX_COMP):
            SbAdd[:, k] = SbAdd[:, k] * C[k]

        StAdd = np.c_[((StDelta - Sbt) / deltaTs).dot(C[:MAX_COMP])]

        Sc = np.hstack([Sbt, SbAdd, StAdd])
        if dictCommonParam['numComp'] == 3:
            Sc[:, 3 + MAX_COMP] = 0
            Sc[:, 3] = 0
        elif dictCommonParam['numComp'] == 2:
            Sc[:, 3 + MAX_COMP] = 0
            Sc[:, 2 + MAX_COMP] = 0
            Sc[:, 3] = 0
            Sc[:, 2] = 0
        elif dictCommonParam['numComp'] == 1:
            Sc[:, 3 + MAX_COMP] = 0
            Sc[:, 2 + MAX_COMP] = 0
            Sc[:, 1 + MAX_COMP] = 0
            Sc[:, 3] = 0
            Sc[:, 2] = 0
            Sc[:, 1] = 0

        while(True):
            flgCalcAll = True
            pinvS = np.linalg.pinv(Sc)
            # pinvS = Sc.T
            C = pinvS.dot(L)

            newBf = Bf.copy()
            for j in range(0, MAX_COMP):
                newBf[j] = Bf[j] + C[MAX_COMP + j]
                if (newBf[j] > maxBf[j]) or (newBf[j] < minBf[j]):
                    Sc[:, MAX_COMP + j] = 0
                    flgCalcAll = False

            newTs = Ts + C[MAX_COMP * 2]
            if abs(newTs) > maxTs:
                Sc[:, MAX_COMP * 2] = 0
                flgCalcAll = False

            if flgCalcAll == True:
                Bf = newBf
                Ts = newTs
                break
            else :
                pass  # Do Nothing

        #print(C)
        #print(Bf)
        #print(Ts)

    Sc = CalcS_Func(dictSignal, dictCommonParam, dictLaserParam, Bf, Ts)
    if dictCommonParam['numComp'] == 3:
        Sc[:, 3] = 0
    elif dictCommonParam['numComp'] == 2:
        Sc[:, 3] = 0
        Sc[:, 2] = 0
    elif dictCommonParam['numComp'] == 1:
        Sc[:, 3] = 0
        Sc[:, 2] = 0
        Sc[:, 1] = 0

    pinvS = np.linalg.pinv(Sc)
    # pinvS = Sc.T
    C = pinvS.dot(L)

    return C

def CorrTemp_Func(dictSignal, LaserParam, C):
    T = dictSignal['T']
    Tstd = LaserParam['Tstd']
    kt = LaserParam['kt']

    return C * (1 + (T - Tstd) * kt)

def CorrLin_Func(LaserParam, C):
    a = LaserParam['aryCoeffLin']
    calcC = 0

    for i in range(0, NUM_COEFF_LIN):
        calcC = calcC + a[i] * pow(C, i)

    return calcC

def CorrZS_Func(LaserParam, C):
    A = LaserParam['coeffA']
    B = LaserParam['coeffB']

    return ((A * C) + B)

# 前処理
np.linalg.pinv(np.full((2,3), 100))
msleep = lambda x: time.sleep(x/1000.0)
np.set_printoptions(linewidth=200)

# 定数
MAX_LASER = 6
MAX_COMP = 5
DATA_COUNT = 500
WIDTH_TS = 0.01
NUM_COEFF_LIN = 5

# 共通パラメータ
dictCommonParam = {'numN':12, 'numNc':5, 'numLaser':1, 'numComp':5}

# 検出器パラメータ
dictDetectorParam = {'kd':0}

# レーザパラメータ
# g_paramS1 = np.loadtxt('S1_inverse.csv', delimiter="\t", dtype='float')
g_paramS1 = np.loadtxt('S.csv', delimiter="\t", dtype='float')
g_paramRz = np.loadtxt('Rz.csv', delimiter="\t", dtype='float')
g_paramRs = np.loadtxt('Rs.csv', delimiter="\t", dtype='float')

lstLaserParam = []
laserParamItem = {'Tstd':25.0, "kt":0, "D0":0, 'lowP':18, 'highP':31, 'deltaBf':0.02, 'deltaTs':0.01, 'maxTs':0.05, 'coeffA':1, 'coeffB':0}
laserParamItem['numTableP'] = laserParamItem['highP'] - laserParamItem['lowP'] + 1
laserParamItem['numTableTs'] = int((laserParamItem['maxTs'] * 2) / WIDTH_TS) + 1
laserParamItem['aryParamS'] = np.reshape(g_paramS1, (laserParamItem['numTableTs'], laserParamItem['numTableP'], dictCommonParam['numN'], MAX_COMP))
laserParamItem['minBf'] = np.full([MAX_COMP], 0.9)
laserParamItem['maxBf'] = np.full([MAX_COMP], 1.2)
laserParamItem['aryCoeffLin'] = [0.0, 1.0, 0.0, 0.0, 0.0]
laserParamItem['aryRz'] = g_paramRz
laserParamItem['aryRs'] = g_paramRs
lstLaserParam.append(laserParamItem)

# 入力信号
dictSignal = {'P':22.0, "T":25.0, "Dof":np.full(500, 0)}

# 入力スペクトル
g_Dz1 = np.loadtxt('Signal.csv', delimiter="\t", usecols=1, dtype='float')
g_Dm1 = np.loadtxt('Signal.csv', delimiter="\t", usecols=2, dtype='float')
lstSpectrum = []
lstSpectrum.append({'aryDz':g_Dz1, "aryDm":g_Dm1})

np.random.seed(0)

TStack = []
TstdStack = []
ktStack = []
spanCorrStack = []
linCorrStack = []
ansStack = []

for lc in range(10):
    print(lc)

    T = 20.0 + 10 * np.random.rand()
    Tstd = 20.0 + 10 * np.random.rand()
    kt = 0.0 + 0.1 * np.random.rand()
    spanCorr = 0.0 + 1.0 * np.random.rand(2)
    linCorr = 0.0 + 1.0 * np.random.rand(5)

    dictSignal['T'] = T
    laserParamItem['Tstd'] = Tstd
    laserParamItem['kt'] = kt
    laserParamItem['coeffA'] = spanCorr[0]
    laserParamItem['coeffB'] = spanCorr[1]
    laserParamItem['aryCoeffLin'] = linCorr

    CalcL = CalcL_Func(dictSignal, dictCommonParam, dictDetectorParam, lstLaserParam[0], lstSpectrum[0])
    CalcC = CLS_Func(dictSignal, dictCommonParam, lstLaserParam[0], CalcL)
    CalcC_Temp = CorrTemp_Func(dictSignal, lstLaserParam[0], CalcC)
    CalcC_Lin = CorrLin_Func(lstLaserParam[0], CalcC_Temp)
    ans = CorrZS_Func(lstLaserParam[0], CalcC_Lin)

    print(CalcC)
    print(CalcC_Temp)
    print(CalcC_Lin)
    print(ans)

    if lc == 0:
        TStack = T
        TstdStack = Tstd
        ktStack = kt
        spanCorrStack = spanCorr
        linCorrStack = linCorr
        ansStack = ans
    else:
        TStack = np.vstack([TStack, T])
        TstdStack = np.vstack([TstdStack, Tstd])
        ktStack = np.vstack([ktStack, kt])
        spanCorrStack = np.vstack([spanCorrStack, spanCorr])
        linCorrStack = np.vstack([linCorrStack, linCorr])
        ansStack = np.vstack([ansStack, ans])

        
np.savetxt('T.csv', TStack, delimiter='\t')
np.savetxt('Tstd.csv', TstdStack, delimiter='\t')
np.savetxt('kt.csv', ktStack, delimiter='\t')
np.savetxt('spanCorr.csv', spanCorrStack, delimiter='\t')
np.savetxt('linCorr.csv', linCorrStack, delimiter='\t')
np.savetxt('ans.csv', ansStack, delimiter='\t')
