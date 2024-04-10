using Microstructure
using Test

axon = Cylinder(da=2.0e-6)
extra = Zeppelin()
iso = Iso()
sphere = Sphere()

bval = [1000, 2500, 5000, 7500, 11100, 18100, 25000, 43000].*1.0e6
techo = 40.0.*ones(8,).*1e-3
tdelta = 15.129.*ones(8,).*1e-3
tsmalldel = 11.0.*ones(8,).*1e-3
prot = Protocol(bval,techo,tdelta,tsmalldel)
    
digits = 6
@test round.(compartment_signals(axon,prot),digits=digits) == round.(
                                            [0.830306256448048,
                                            0.660977107327415,
                                            0.500413251789382,
                                            0.411543391237258,
                                            0.336884386133270,
                                            0.260507095967021,
                                            0.218862593318336,
                                            0.161439844983240],digits=digits)

@test round.(compartment_signals(extra,prot),digits=digits) == round.(
                                            [0.672953994843349,
                                            0.376716014811867,
                                            0.148013602779966,
                                            0.060161061844622,
                                            0.017211501723351,
                                            0.001665325091163,
                                            1.789612484149176e-04,
                                            6.163836418812522e-07],digits=digits)

@test round.(compartment_signals(iso,prot),digits=digits)  == round.(
                                            [0.135335283236613,
                                            0.006737946999085,
                                            4.539992976248477e-05,
                                            3.059023205018258e-07,
                                            2.283823312361578e-10,
                                            1.899064673586898e-16,
                                            1.928749847963932e-22,
                                            4.473779306181057e-38],digits=digits)

@test round.(compartment_signals(sphere,prot),digits=digits) == round.(
                                            [0.926383765355293,
                                            0.825994848716073,
                                            0.682267490105489,
                                            0.563549432273578,
                                            0.427936553427585,
                                            0.250562419120995,
                                            0.147833680281184,
                                            0.0373258948718356],digits=digits)