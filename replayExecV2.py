#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os
import glob
import matplotlib.pyplot as plt
import pdb

def simulate_dynamics(y_true, insulin, meal, basal, params, meanB):
    """
    Simulate glucose dynamics given inputs and parameter set.
    Uses updated tau and initial conditions.
    """
    kempt = params['kempt']
    kabs  = params['kabs']
    f     = params['f']
    Gb    = params['Gb']
    SG    = params['SG']
    Vg    = params['Vg']
    p2    = params['p2']
    SI    = params['SI']
    alpha = params['alpha']
    kd    = params['kd']
    ka2   = params['ka2']
    ke    = params['ke']
    Vi    = params['Vi']
    r2    = params['r2']

    n = len(y_true)
    GVal   = np.zeros(n)
    gVal   = np.zeros(n)
    Isc1   = np.zeros(n)
    Isc2   = np.zeros(n)
    Ip     = np.zeros(n)
    Qsto1  = np.zeros(n)
    Qsto2  = np.zeros(n)
    Qgut   = np.zeros(n)
    xVal   = np.zeros(n)

    # initialize states
    GVal[0]  = y_true[0]
    gVal[0]  = y_true[0]
    Isc1[0]  = basal[0] / (kd*Vi)
    Isc2[0]  = kd*Isc1[0] / ka2
    Ip[0]    = ka2*Isc2[0] / ke
    tau = 5
    #pdb.set_trace()
    for i in range(1, n):
        prevG = GVal[i-1]
        D = 1.0 if (prevG >= 60.0 and prevG < 119.13) else 0.0
        E = 1.0 if (prevG < 60.0) else 0.0
        risk = 1+(
            10 * ((np.log(np.maximum(prevG,25))**r2 - np.log(119.13)**r2)**2) * D
            + 10 * ((np.log(60)**r2 - np.log(119.13)**r2)**2) * E
        )
       
        #risk = 0.1
        risk = np.abs(risk)        
        Ipb = Ip[0]

        # update dynamics (tau multiplier)
        dummyIsc1 = Isc1[i-1] + tau * (-kd * Isc1[i-1] + (basal[i-1] + insulin[i-1]) / Vi)
        dummyIsc2 = Isc2[i-1] + tau * (kd * Isc1[i-1] - ka2 * Isc2[i-1])
        dummyIp   = Ip[i-1]   + tau * (ka2 * Isc2[i-1] - ke * Ip[i-1])
        dummyQsto1= Qsto1[i-1]+ tau * (-kempt * Qsto1[i-1] + meal[i-1])
        #dummyQsto2= Qsto2[i-1]+ tau * (kempt * Qsto1[i-1] - kempt * Qsto2[i-1])
        dummyQsto2= Qsto2[i-1]+ tau * (kempt * dummyQsto1 - kempt * Qsto2[i-1])
        #dummyQgut = Qgut[i-1] + tau * (kempt * Qsto2[i-1] - kabs * Qgut[i-1])
        dummyQgut = Qgut[i-1] + tau * (kempt * dummyQsto2 - kabs * Qgut[i-1])

        dummyXVal = xVal[i-1] + tau * (-p2 * xVal[i-1] - SI * (Ip[i-1] - Ipb))
        #dummygVal = (
        #    gVal[i-1] + tau * (-(SG + risk * xVal[i-1]) * gVal[i-1] + SG * Gb + f * kabs * Qgut[i-1] / Vg)
        #)


        dummygVal = (
            gVal[i-1] + tau * (-(SG + risk * xVal[i-1]) * gVal[i-1] + SG * Gb + f * kabs * dummyQgut / Vg)
        )

        dummyG1 = GVal[i-1] + tau * (-(1/alpha) * (GVal[i-1] - dummygVal))
        if ((dummyG1 - GVal[i-1])**2).sum()/256 - 100000.0 > 0:
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

    #pdb.set_trace()
    return GVal


def lossF(y_true, GVal):
    """RMSE over the whole series."""
    return np.sqrt(np.mean((y_true - GVal)**2))


def lossFV2(y_true, GVal):
    """MARD (Mean Absolute Relative Difference) in %."""
    mask = y_true != 0
    return 100.0 * np.mean(np.abs(y_true[mask] - GVal[mask]) / y_true[mask])


def main():
    parser = argparse.ArgumentParser(description="Simulate multiple scenarios for one patient")
    parser.add_argument('--scene_folder', required=True, help="folder containing scenario CSVs")
    parser.add_argument('--constants',    required=True, help="CSV with numID and parameters")
    parser.add_argument('--id',           type=int, default=1, help="patient numID")
    args = parser.parse_args()
    
    const = pd.read_csv(args.constants)
    row = const[const['numID']==args.id]
    if row.empty:
        raise ValueError(f"numID {args.id} not found in constants file")
    row = row.iloc[0]
    params = {k: row[k] for k in ['kempt','kabs','f','Gb','SG','Vg','p2','SI','alpha','kd','ka2','ke','Vi','r2']}

    pattern = os.path.join(args.scene_folder, '*.csv')
    files = sorted(glob.glob(pattern))
    scen_files = [f for f in files if os.path.basename(f).lower().startswith(f'p{args.id}_')]
    #scen_files = scen_files[-7:]
    if not scen_files:
        raise ValueError(f"No scenario files for patient {args.id} in {args.scene_folder}")

    rmses, mards, x = [], [], []
    plt.figure(figsize=(10,6))
    for scen_file in scen_files:
        scen = pd.read_csv(scen_file)
        y_true  = scen['glucose'].to_numpy(dtype=float)
        insulin = scen['bolus'  ].to_numpy(dtype=float) * 1000/(76.37*5)
        meal    = scen['cho'    ].to_numpy(dtype=float) * 1000/(76.37*5)
        basal   = scen['basal'  ].to_numpy(dtype=float) *1000/76.37
        
        base, _ = os.path.splitext(scen_file)

        # 2) split on "_" and take the last piece
        last_piece = base.split("_")[-1]

        # 3) convert to int
        last_number = int(last_piece)
        # downsample by 5
        y_true  = y_true[::5]
        insulin = insulin[::5]
        meal    = meal[::5]
        basal   = basal[::5]
        meanB = np.mean(basal) 
        pdb.set_trace()
        GVal = simulate_dynamics(y_true, insulin, meal, basal, params, meanB)
        rmse = lossF(y_true, GVal)
        mard = lossFV2(y_true, GVal)
        rmses.append(rmse)
        mards.append(mard)
        x.append(last_number)
        label = os.path.basename(scen_file)
        t = np.arange(len(y_true))
    plt.scatter(x, rmses)
    plt.scatter(x, mards)

    plt.xlabel('Scenarios')
    plt.ylabel('RMSE')
    plt.title(f'Patient {args.id}: True vs Predicted Across Scenarios')
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()

    #print(f"Patient {args.id} over {len(scen_files)} scenarios:")
    #print(f"  RMSE: mean={rmses:.4f}")
    #print(f"  MARD: mean={mards:.2f}%")

if __name__=='__main__':
    main()
