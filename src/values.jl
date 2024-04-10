# Constants used in modelling
#
# SI units are used in computations but we need to pay attention to some conventions between text files and computation
# Units used: s (second), m (meter), T (tesla), s/m^2, m^2/s, T/m
#
# In text files, e.g. bval, techo, tdelta and tsmalldel, units are s/mm^2 for b-values and ms for times. 
# Protocols reading from text files will automatically convert units based on the convention. 
#
# In ploting or visualize microstructure parameters, ms (t2),um (size),and um^2/ms (diffusivity) are used.
#
# b-values: 1 ms/um^2 = 10^3 s/mm^2 = 10^9 s/m^2
# G: 1000 mT/m = 1 T/m
# diffusivity: 1 um^2/ms = 10^-9 m^2/s

using StaticArrays

"""
gyromagnetic ratio
"""
const gmr = 2.67*1e8 #rad/s/T

# Up to 10th order
const BesselJ_RootsCylinder = @SVector [ 
1.84118378134066
5.33144277352503
8.53631636634629
11.7060049025921
14.8635886339090
18.0155278626818
21.1643698591888
24.3113268572108
27.4570505710592
30.6019229726691]

# Calculated from camino function BesselJ_RootsSphere
# Up to 31st
const BesselJ_RootsSphere = @SVector [
2.0815759778181
5.94036999057271
9.20584014293667
12.404445021902
15.5792364103872
18.7426455847748
21.8996964794928
25.052825280993
28.2033610039524
31.3520917265645
34.499514921367
37.6459603230864
40.7916552312719
43.9367614714198
47.0813974121542
50.2256516491831
53.3695918204908
56.5132704621986
59.6567290035279
62.8000005565198
65.9431119046553
69.0860849466452
72.2289377620154
75.3716854092873
78.5143405319308
81.6569138240367
84.7994143922025
87.9418500396598
91.0842274914688
94.2265525745684
97.368830362901]

# scaling factors to tissue parameters
# scaling_factors 