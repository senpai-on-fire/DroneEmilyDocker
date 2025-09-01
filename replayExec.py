#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import pdb
import matplotlib.pyplot as plt
def simulate_dynamics(y_true, insulin, meal, basal, params):
    """
    Simulate glucose dynamics given inputs and parameter set.
    Returns predicted glucose time‐series GVal.
    """
    kempt = params['kempt']
    kabs  = params['kabs']
    f     = params['f']
    Gb    = params['Gb']
    SG    = params['SG']
    Vg    = params['Vg']
    p2    = params['p2']
    SI    = params['SI']
    alpha= params['alpha']
    kd    = params['kd']
    ka2   = params['ka2']
    ke    = params['ke']
    Vi    = params['Vi']
    r2    = params['r2']

    n = len(y_true)
    # state arrays
    GVal   = np.zeros(n)
    gVal   = np.zeros(n)
    Isc1   = np.zeros(n)
    Isc2   = np.zeros(n)
    Ip     = np.zeros(n)
    Qsto1  = np.zeros(n)
    Qsto2  = np.zeros(n)
    Qgut   = np.zeros(n)
    xVal   = np.zeros(n)

    # initialize
    GVal[0] = y_true[0]
    gVal[0] = y_true[0]
    Isc1[0] = 1.22 / kd
    Isc2[0] = 1.22 / ka2
    Ip[0] = 1.22 / ke
    tau = 5
    pdb.set_trace()
    for i in range(1, n):
        # effective plasma insulin (from bolus input)
        
        Ipb = 1.22/ke
        prevG = GVal[i-1]
        D = 1.0 if (prevG >= 60.0 and prevG < 119.13) else 0.0
        E = 1.0 if (prevG < 60.0) else 0.0
        risk = (
            10 * ((np.log(prevG)**r2 - np.log(119.13)**r2)**2) * D
            + 10 * ((np.log(60)**r2 - np.log(119.13)**r2)**2) * E
        )

        # update equations (tau = 1)
        dummyIsc1 = Isc1[i-1] + tau*(-kd * Isc1[i-1] + (basal[i-1] + insulin[i-1]) / Vi)
        dummyIsc2 = Isc2[i-1] + tau*(kd * Isc1[i-1] - ka2 * Isc2[i-1])
        dummyIp   = Ip[i-1]   + tau*(ka2 * Isc2[i-1] - ke * Ip[i-1])
        dummyQsto1= Qsto1[i-1]+ tau*(-kempt * Qsto1[i-1] + meal[i-1])
        dummyQsto2= Qsto2[i-1]+ tau*(kempt * Qsto1[i-1] - kempt * Qsto2[i-1])
        dummyQgut = Qgut[i-1] + tau*(kempt * Qsto2[i-1] - kabs * Qgut[i-1])
        dummyXVal = xVal[i-1] + tau*(-p2 * xVal[i-1] - SI * (Ip[i-1] - Ipb))
        dummygVal = (
            gVal[i-1] 
            + tau*(-(SG + risk * xVal[i-1]) * gVal[i-1] + SG * Gb + f * kabs * Qgut[i-1] / Vg)
        )

        dummyG1 = GVal[i-1] + tau*(-(1/alpha) * (GVal[i-1] - gVal[i-1]))
        if ( (dummyG1 - GVal[i-1])**2 ).sum()/256 - 100000.0 > 0:
            dummyG1 = GVal[i-1]

        # commit updates
        Isc1[i]  = dummyIsc1
        Isc2[i]  = dummyIsc2
        Ip[i]    = dummyIp
        Qsto1[i] = dummyQsto1
        Qsto2[i] = dummyQsto2
        Qgut[i]  = dummyQgut
        xVal[i]  = dummyXVal
        gVal[i]  = dummygVal
        GVal[i]  = dummyG1

    return GVal

def lossF(y_true, GVal):
    """RMSE over the whole series."""
    return np.sqrt(np.mean((y_true - GVal)**2))

def lossFV2(y_true, GVal):
    """MARD (Mean Absolute Relative Difference) in %."""
    mask = y_true != 0
    return 100.0 * np.mean(np.abs(y_true[mask] - GVal[mask]) / y_true[mask])

def main():
    p = argparse.ArgumentParser(description="Run lossF and lossFV2 on a patient scenario")
    p.add_argument('--scenario',  required=True, help="CSV with t,glucose,cho,bolus,basal,…")
    p.add_argument('--constants', required=True, help="CSV with numID and parameter columns")
    p.add_argument('--id',        type=int, default=1, help="Which numID to pick")
    args = p.parse_args()

    # load data
    scen  = pd.read_csv(args.scenario)
    const = pd.read_csv(args.constants)
    row   = const[const['numID']==args.id]
    if row.empty:
        raise ValueError(f"numID {args.id} not found in constants file")
    row = row.iloc[0]

    params = {
        'kempt': row.kempt, 'kabs': row.kabs, 'f': row.f, 'Gb': row.Gb, 'SG': row.SG,
        'Vg': row.Vg, 'p2': row.p2, 'SI': row.SI, 'alpha': row.alpha, 'kd': row.kd,
        'ka2': row.ka2, 'ke': row.ke, 'Vi': row.Vi, 'r2': row.r2
    }
    
    y_true = scen['glucose'].to_numpy(dtype=float)
    insulin = scen['bolus'].to_numpy(dtype=float)*100*1000/(76.37*5)
    meal   = scen['cho'].to_numpy(dtype=float)*1000/(76.37*5)
    basal  = scen['basal'].to_numpy(dtype=float)*100*1000/76.37
    
    y_true = y_true[::5]
    insulin = insulin[::5]
    maeal = meal[::5]
    basal = basal[::5]
    pdb.set_trace()
    GVal = simulate_dynamics(y_true, insulin, meal, basal, params)
    
    plt.figure(figsize=(10,5))
    plt.plot(y_true,label='True Glucose')
    plt.plot(GVal,label='Replay Glucose')
    plt.show()
    rmse = lossF(y_true, GVal)
    mard = lossFV2(y_true, GVal)

    print(f"Results for numID={args.id}:")
    print(f"  • lossF  (RMSE) : {rmse:.4f}")
    print(f"  • lossFV2 (MARD): {mard:.2f}%")

if __name__=='__main__':
    main()

