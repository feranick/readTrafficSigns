#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************************************
* Augment Images using ImgAug
*****************************************************************************
'''
print(__doc__)

import os, sys, glob, cv2
from imgaug import augmenters as iaa
import pandas as pd

#***************************************************
# This is needed for installation through pip
#***************************************************
def AugmentImages():
    main()

#************************************
# Main
#************************************
def main():
    r1 = -45
    r2 = 45
    s1 = -10
    s2 = 10
    n = 10
    CSVfile = 'Train.csv'

    df = pd.read_csv(CSVfile)
    print(df.to_string())
    df2 = df.copy()
    print(len(df.index))

    for i, row in df.iterrows():
        file = df['Path'][i]
        print(" Opening:",file)
        for i in range(1,n):
            image = cv2.imread(file)
            rotate = iaa.Affine(rotate=(r1, r2))
            iaa.Affine(shear=(s1, s2)) # shear by -10 to +10 degrees
            image_aug = rotate(image=image)
            cv2.imwrite(os.path.splitext(file)[0]+"_"+str(i)+".png", image_aug)
            row[7] = os.path.splitext(file)[0]+"_"+str(i)+".png"
            df2.loc[len(df2.index)] = row
                
        print(" #"+str(n)+" additional augmented data from file:"+file)
    
    print(df2.to_string())
    df2.to_csv(os.path.splitext(CSVfile)[0]+"_augmented.csv", index=False)
    print(" \nSaved to: "+os.path.splitext(CSVfile)[0]+"_augmented.csv\n")
    
#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())




