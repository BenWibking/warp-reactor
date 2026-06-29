# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: Mojo port of the standalone primordial chemistry ROS2S reproducer.

from layout import Layout, LayoutTensor
from std.math import abs, cbrt, exp, log, max, min, sqrt
from std.sys import argv

comptime NumSpec = 14
comptime Neqs = NumSpec + 1
comptime NetIenuc = NumSpec
comptime DefaultRedshift = 30.0
comptime GravConstant = 6.674e-8
comptime Pi = 3.141592653589793238462643383279502884
comptime Log10 = 2.3025850929940456840179914546843642
comptime NA = 6.02214076e23
comptime KB = 1.3806490000000002e-16
comptime MP = 1.67262192595e-24

comptime Success = 1
comptime BadInputs = -1
comptime DtUnderflow = -2
comptime TooManySteps = -4
comptime TooMuchAccuracyRequested = -5
comptime LuDecompositionError = -7

comptime TffReduc = 1.0e-1
comptime MaxCollapseSteps = 1000
comptime InitialTemperature = 1.0e2
comptime RtolSpec = 1.0e-4
comptime AtolSpec = 1.0e-4
comptime RtolEnergy = 1.0e-6
comptime AtolEnergy = 1.0e-6
comptime DefaultGridDim = 64
comptime PerturbationInterval = 20
comptime PerturbationAmplitude = 0.2

comptime SpeciesLayout = Layout.row_major(NumSpec)
comptime VecLayout = Layout.row_major(Neqs)
comptime MatLayout = Layout.row_major(Neqs, Neqs)
comptime SpeciesTensor = LayoutTensor[DType.float64, SpeciesLayout, MutAnyOrigin]
comptime VecTensor = LayoutTensor[DType.float64, VecLayout, MutAnyOrigin]
comptime MatTensor = LayoutTensor[DType.float64, MatLayout, MutAnyOrigin]


def truthy(x: Bool) -> Bool:
    return x


def truthy(x: Float64) -> Bool:
    return x != 0.0


def powi_m3(x: Float64) -> Float64:
    return 1.0 / (x * (x * x))


def powi_m5(x: Float64) -> Float64:
    return 1.0 / (x * ((x * x) * (x * x)))


def powi_m7(x: Float64) -> Float64:
    var x3 = x * (x * x)
    return 1.0 / (x * (x3 * x3))


def vget(v: SpeciesTensor, i: Int) -> Float64:
    return rebind[Scalar[DType.float64]](v[i])


def vget(v: VecTensor, i: Int) -> Float64:
    return rebind[Scalar[DType.float64]](v[i])


def vset(mut v: SpeciesTensor, i: Int, value: Float64):
    v[i] = value


def vset(mut v: VecTensor, i: Int, value: Float64):
    v[i] = value


def mget(m: MatTensor, i: Int, j: Int) -> Float64:
    return rebind[Scalar[DType.float64]](m[i, j])


def mset(mut m: MatTensor, i: Int, j: Int, value: Float64):
    m[i, j] = value


struct BurnState(Copyable, Movable):
    var rho: Float64
    var T: Float64
    var e: Float64
    var xn: InlineArray[Float64, NumSpec]

    def __init__(out self):
        self.rho = 0.0
        self.T = 0.0
        self.e = 0.0
        self.xn = InlineArray[Float64, NumSpec](fill=0.0)


struct EosSums(Copyable, Movable):
    var sum_abarinv: Float64
    var sum_gammasinv: Float64
    var gasconstant: Float64

    def __init__(out self, a: Float64, g: Float64, gas: Float64):
        self.sum_abarinv = a
        self.sum_gammasinv = g
        self.gasconstant = gas


struct IntegratorStats(Copyable, Movable):
    var internal_steps: Int
    var rhs_calls: Int
    var jacobian_calls: Int
    var decompositions: Int
    var linear_solves: Int
    var accepted_steps: Int
    var rejected_steps: Int

    def __init__(out self):
        self.internal_steps = 0
        self.rhs_calls = 0
        self.jacobian_calls = 0
        self.decompositions = 0
        self.linear_solves = 0
        self.accepted_steps = 0
        self.rejected_steps = 0


struct CollapseState(Copyable, Movable):
    var current: BurnState
    var time: Float64
    var density_driver: Float64
    var completed_steps: Int
    var stats: IntegratorStats

    def __init__(out self):
        self.current = BurnState()
        self.time = 0.0
        self.density_driver = 0.0
        self.completed_steps = 0
        self.stats = IntegratorStats()


struct Options(Copyable, Movable):
    var grid_dim: Int
    var perturb: Bool
    var show_help: Bool

    def __init__(out self):
        self.grid_dim = DefaultGridDim
        self.perturb = True
        self.show_help = False


def species_mass(n: Int) -> Float64:
    if n == 0:
        return 9.10938188e-28
    if n == 1:
        return 1.67262158e-24
    if n == 2:
        return 1.67353251819e-24
    if n == 3:
        return 1.67444345638e-24
    if n == 4:
        return 3.34512158e-24
    if n == 5:
        return 3.34603251819e-24
    if n == 6:
        return 3.34615409819e-24
    if n == 7:
        return 3.34694345638e-24
    if n == 8:
        return 3.34706503638e-24
    if n == 9:
        return 5.01865409819e-24
    if n == 10:
        return 5.01956503638e-24
    if n == 11:
        return 6.69024316e-24
    if n == 12:
        return 6.69115409819e-24
    return 6.69206503638e-24


def species_gamma(n: Int) -> Float64:
    if n == 6 or n == 8 or n == 9 or n == 10:
        return 1.4
    return 5.0 / 3.0


def small_number_density_floor() -> Float64:
    return 1.0e-100


def redshift() -> Float64:
    return DefaultRedshift


def density(xn: InlineArray[Float64, NumSpec]) -> Float64:
    var rho = 0.0
    for n in range(NumSpec):
        rho += xn[n] * species_mass(n)
    return rho


def eos_sums_from_number_densities(xn: InlineArray[Float64, NumSpec]) -> EosSums:
    var gasconstant = NA * KB
    var protonmass = MP
    var sum_abarinv = 0.0
    var sum_gammasinv = 0.0
    var rhotot = 0.0
    for n in range(NumSpec):
        rhotot += xn[n] * species_mass(n)
    for n in range(NumSpec):
        sum_abarinv += xn[n]
        sum_gammasinv += (xn[n] * protonmass / rhotot) * (1.0 / (species_gamma(n) - 1.0))
    sum_abarinv *= protonmass / rhotot
    sum_gammasinv /= sum_abarinv
    return EosSums(sum_abarinv, sum_gammasinv, gasconstant)


def eos_rt(mut state: BurnState):
    var sums = eos_sums_from_number_densities(state.xn)
    state.e = sums.sum_gammasinv * sums.sum_abarinv * sums.gasconstant * state.T


def eos_re(mut state: BurnState):
    var sums = eos_sums_from_number_densities(state.xn)
    state.T = state.e / (sums.sum_gammasinv * sums.gasconstant * sums.sum_abarinv)


def balance_charge(mut state: BurnState):
    state.xn[0] = -state.xn[3] - state.xn[7] + state.xn[1] + state.xn[12] + state.xn[6] + state.xn[4] + state.xn[9] + 2.0 * state.xn[11]


def normalize_number_densities_to_density(mut state: BurnState):
    var mass_fractions = InlineArray[Float64, NumSpec](fill=0.0)
    var total = 0.0
    for n in range(NumSpec):
        mass_fractions[n] = species_mass(n) * state.xn[n] / state.rho
        total += mass_fractions[n]
    for n in range(NumSpec):
        mass_fractions[n] /= total
        state.xn[n] = mass_fractions[n] * state.rho / species_mass(n)


def floor_and_normalize_number_densities(mut state: BurnState):
    for n in range(NumSpec):
        state.xn[n] = max(state.xn[n], small_number_density_floor())
    normalize_number_densities_to_density(state)


def rhs_specie(state: BurnState, mut ydot: VecTensor, X: SpeciesTensor, z: Float64):
    var T = state.T
    var x0_0 = exp((-0.75)*log(abs(T)))
    var x1_0 = 2.5950363272655348e-10*vget(X, 0)*vget(X, 4)*x0_0
    var x2_0 = 1.3300135414628029e-18*exp((0.94999999999999996)*log(abs(T)))*vget(X, 0)*vget(X, 5)*exp(-0.00010729613733905579*T)
    var x3_0 = ((T)*(T))
    var x4_0 = T <= 10000.0
    var x5_0 = vget(X, 0)*vget(X, 6)*((((
       -5.5279999999999998e-28*((T)*(T)*(T)*(T)*(T)) + 3.3467999999999999e-23*((((T)*(T)))*(((T)*(T)))) - 7.5474000000000004e-19*((T)*(T)*(T)) - 2.3088e-11*T + 7.3427999999999993e-15*x3_0 + 4.2277999999999996e-8
    )) if truthy((x4_0)) else ((
       0
    ))))
    var x6_0 = log(T)
    var x7_0 = 0.10684732509875319*x6_0 - 1
    var x8_0 = ((x7_0)*(x7_0))
    var x9_0 = ((x7_0)*(x7_0)*(x7_0))
    var x10_0 = ((((x7_0)*(x7_0)))*(((x7_0)*(x7_0))))
    var x11_0 = ((x7_0)*(x7_0)*(x7_0)*(x7_0)*(x7_0))
    var x12_0 = exp((6)*log(abs(x7_0)))
    var x13_0 = ((x7_0)*(x7_0)*(x7_0)*(x7_0)*(x7_0)*(x7_0)*(x7_0))
    var x14_0 = exp((8)*log(abs(x7_0)))
    var x15_0 = ((x7_0)*(x7_0)*(x7_0)*(x7_0)*(x7_0)*(x7_0)*(x7_0)*(x7_0)*(x7_0))
    var x16_0 = vget(X, 2)*vget(X, 3)
    var x17_0 = x16_0*((((
       1.4643482606109061e-16*exp((1.78186)*log(abs(T)))
    )) if truthy((T <= 1160.0)) else ((
       3.3178155742407614e-14*exp((1.1394493358416311)*log(abs(T)))*exp(-10.993097527150175*x10_0 + 14.449862906216714*x11_0 + 58.228375789703179*x12_0 - 162.59852239006702*x13_0 + 144.55426734953477*x14_0 - 44.454280878123605*x15_0 - 12.447178055372778*x8_0 + 6.9391784778399117*x9_0)
    ))))
    var x18_0 = vget(X, 1)*vget(X, 3)
    var x19_0 = 1.0e-8*exp((-0.40000000000000002)*log(abs(T)))*x18_0
    var x20_0 = 2.6534040307116387e-9*exp((-0.10000000000000001)*log(abs(T)))
    var x21_0 = vget(X, 3)*vget(X, 5)
    var x22_0 = x20_0*x21_0
    var x23_0 = vget(X, 0)*vget(X, 2)
    var x24_0 = 1.4000000000000001e-18*exp((0.92800000000000005)*log(abs(T)))*x23_0*exp(-6.1728395061728397e-5*T)
    var x25_0 = vget(X, 0)*vget(X, 12)
    var x26_0 = 3.8571873359681582e-209*exp((43.933476326349997)*log(abs(T)))*x25_0*exp(-5902.1601240760483*x10_0 + 5825.9326359379538*x11_0 - 3578.1439181805954*x12_0 + 1242.7294446825149*x13_0 - 186.35635455381879*x14_0 - 1618.789587733125*x8_0 + 3854.4033653120223*x9_0)
    var x27_0 = 3.7903999274394518e-18*exp((2.360852208681)*log(abs(T)))*vget(X, 0)*vget(X, 3)*exp(-258.18559308467115*x10_0 + 846.15238706523724*x11_0 - 1113.0879095147111*x12_0 + 671.95094388835207*x13_0 - 154.90262957142161*x14_0 - 24.766609674457612*x8_0 + 13.307984239358756*x9_0)
    var x28_0 = sqrt(T)
    var x29_0 = 1.0/x28_0
    var x30_0 = vget(X, 0)*x29_0
    var x31_0 = 5.7884371785482823e-10*vget(X, 11)*x30_0*exp((-1.7524)*log(abs(0.00060040841663220993*x28_0 + 1.0)))*exp((-0.24759999999999999)*log(abs(0.32668576019240059*x28_0 + 1.0)))
    var x32_0 = 1.4981088130721367e-10*exp((-0.63529999999999998)*log(abs(T)))
    var x33_0 = 8.6173430000000006e-5*T
    var x34_0 = 1.0/T
    var x35_0 = -4.3524079114767552e-117*exp((23.915965629999999)*log(abs(T)))*vget(X, 0)*vget(X, 13)*exp(-4361.9927099007555*x10_0 + 4879.7345146260486*x11_0 - 3366.4639698826941*x12_0 + 1300.3028484326148*x13_0 - 214.82451513312137*x14_0 - 941.91483008144996*x8_0 + 2506.9866529060901*x9_0) + x25_0*((((
       x32_0
    )) if truthy((x33_0 <= 9280.0)) else ((
       1250086.112245841*exp((-1.5)*log(abs(T)))*(1.5400000000000001e-9 + 4.6200000000000001e-10*exp(-93988.701501924661*x34_0))*exp(-469943.50750964211*x34_0) + x32_0
    ))))
    var x36_0 = -5.9082438637265071e-70*exp((13.536555999999999)*log(abs(T)))*x23_0*exp(-2207.4643501257692*x10_0 + 2500.8077583366976*x11_0 - 1768.8867461266502*x12_0 + 704.19926629500367*x13_0 - 120.0438480494693*x14_0 - 502.72883252679094*x8_0 + 1281.477767828706*x9_0) + vget(X, 0)*vget(X, 1)*((((
       x32_0
    )) if truthy((x33_0 <= 5500.0)) else ((
       3.2867337024382687e-10*exp((-0.72411256578268512)*log(abs(T)))*exp(-2.4649195146505534*x10_0 - 1.020773727011937*x11_0 + 3.3530579587656564*x12_0 + 3.6203127646377791*x13_0 - 1.0930705283186732*x14_0 - 1.6921001126637107*x15_0 - 1.774686809424741*x8_0 - 1.951835616513679*x9_0)
    ))))
    var x37_0 = 7.1999999999999996e-8*vget(X, 9)*x30_0
    var x38_0 = vget(X, 2)*vget(X, 7)
    var x39_0 = x20_0*x38_0
    var x40_0 = x16_0*(1.3500000000000001e-9*exp((0.098492999999999997)*log(abs(T))) + 4.4350199999999998e-10*exp((0.55610000000000004)*log(abs(T))) + 3.7408500000000004e-16*exp((2.1825999999999999)*log(abs(T))))/(0.0061910000000000003*exp((1.0461)*log(abs(T))) + 8.9711999999999997e-11*exp((3.0424000000000002)*log(abs(T))) + 3.2575999999999999e-14*exp((3.7740999999999998)*log(abs(T))) + 1.0)
    var x41_0 = vget(X, 0)*vget(X, 8)
    var x42_0 = 35.5*exp((-2.2799999999999998)*log(abs(T)))*x41_0*exp(-46707.0*x34_0)
    var x43_0 = x37_0 - x39_0 - x40_0 + x42_0
    var x44_0 = Log10
    var x45_0 = 1.0/x44_0
    var x46_0 = x45_0*x6_0
    var x47_0 = exp((-0.12690000000000001*exp((-3.0)*log(abs(x44_0)))*((x6_0)*(x6_0)*(x6_0)) + 1.1180000000000001*exp((-2.0)*log(abs(x44_0)))*((x6_0)*(x6_0)) - 1.5229999999999999*x46_0 - 19.379999999999999)*log(abs(10.0)))
    var x48_0 = vget(X, 1)*vget(X, 5)
    var x49_0 = x47_0*x48_0
    var x50_0 = x18_0*((((
       -7.7700000000000002e-13*T + 2.5000000000000002e-10*x28_0 + 2.96e-6*x29_0 - 1.73e-9
    )) if truthy((T >= 10.0  and  T <= 100000.0)) else ((
       0
    ))))
    var x51_0 = 1.0000000000000001e-9*vget(X, 1)*vget(X, 10)*exp(-457.0*x34_0)
    var x52_0 = ((x6_0)*(x6_0))
    var x53_0 = exp((-2)*log(abs(x44_0)))
    var x54_0 = x52_0*x53_0
    var x55_0 = 8.4600000000000008e-10*x46_0 - 1.3700000000000002e-10*x54_0 + 4.1700000000000001e-10
    var x56_0 = ((x6_0)*(x6_0)*(x6_0))
    var x57_0 = x56_0/((x44_0)*(x44_0)*(x44_0))
    var x58_0 = vget(X, 1)*vget(X, 2)*((((
       3.4977396723747635e-20*exp((-0.14999999999999999)*log(abs(T)))
    )) if truthy((T < 30)) else ((
       exp((-3.194*x46_0 + 1.786*x52_0*x53_0 - 0.2072*x57_0 - 18.199999999999999)*log(abs(10.0)))
    ))))
    var x59_0 = 6.0e-10*vget(X, 2)*vget(X, 6)
    var x60_0 = ((((x6_0)*(x6_0)))*(((x6_0)*(x6_0))))
    var x61_0 = ((x6_0)*(x6_0)*(x6_0)*(x6_0)*(x6_0))
    var x62_0 = vget(X, 1)*vget(X, 8)*((((
       (-1.4491368e-7*x52_0 + 3.4172804999999998e-8*x56_0 + 3.5311931999999998e-13*((x6_0)*(x6_0)*(x6_0)*(x6_0)*(x6_0)*(x6_0)*(x6_0)) - 1.8171411000000001e-11*exp((6)*log(abs(x6_0))) + 3.3735381999999997e-7*x6_0 - 4.7813727999999997e-9*x60_0 + 3.9731542e-10*x61_0 - 3.3232183000000002e-7)*exp(-21237.150000000001*x34_0)
    )) if truthy((T >= 100.0  and  T <= 30000.0)) else ((
       0
    ))))
    var x63_0 = -x59_0 + x62_0
    var x64_0 = x19_0 + x58_0 + x63_0
    var x65_0 = 6.3999999999999996e-10*vget(X, 2)*vget(X, 9)
    var x66_0 = -x65_0
    var x67_0 = exp((-0.5)*log(abs(T)))
    var x68_0 = vget(X, 7)*x67_0
    var x69_0 = 7.9674337148168363e-7*vget(X, 1)*x68_0
    var x70_0 = vget(X, 1)*vget(X, 13)*((((
       1.26e-9*x0_0*exp(-127500.0*x34_0)
    )) if truthy((x4_0)) else ((
       4.0000000000000003e-37*exp((4.7400000000000002)*log(abs(T)))
    ))))
    var x71_0 = exp((0.25)*log(abs(T)))
    var x72_0 = 2.8833736969617052e-16*vget(X, 12)*vget(X, 2)*x71_0
    var x73_0 = T >= 50.0
    var x74_0 = x48_0*((((
       2.0000000000000001e-10*exp((0.40200000000000002)*log(abs(T)))*exp(-37.100000000000001*x34_0) - 3.3099999999999998e-17*exp((1.48)*log(abs(T)))
    )) if truthy((x73_0)) else ((
       0
    ))))
    var x75_0 = vget(X, 2)*vget(X, 4)
    var x76_0 = x75_0*((((
       2.0299999999999998e-9*exp((-0.33200000000000002)*log(abs(T))) + 2.0600000000000001e-10*exp((0.39600000000000002)*log(abs(T)))*exp(-33.0*x34_0)
    )) if truthy((x73_0)) else ((
       0
    ))))
    var x77_0 = x74_0 - x76_0
    var x78_0 = x36_0 + x66_0 + x69_0 + x70_0 - x72_0 + x77_0
    var x79_0 = 4.9999999999999996e-6*vget(X, 3)*vget(X, 6)*x29_0
    var x80_0 = ((vget(X, 2))*(vget(X, 2))*(vget(X, 2)))
    var x81_0 = 1.0/x71_0
    var x82_0 = 1.0e-25*vget(X, 2)*vget(X, 5)
    var x83_0 = vget(X, 1) + vget(X, 10) + vget(X, 2) + vget(X, 3) + 2.0*vget(X, 6) + 2.0*vget(X, 8) + vget(X, 9)
    var x84_0 = -133.82830000000001*x34_0 - 4.8909149999999997*x45_0*x6_0 + 0.47490300000000002*x54_0
    var x85_0 = exp(-0.0022727272727272726*T)
    var x86_0 = exp(-0.00054054054054054055*T)
    var x87_0 = -2.0563129999999998*x85_0 + 0.58640729999999996*x86_0 + 0.82274429999999998
    var x88_0 = -69.700860000000006*x45_0*log(40870.379999999997*x34_0 + 1.0) + 4.6331670000000003*x57_0
    var x89_0 = exp((-23705.700000000001*x34_0 - 2080.4099999999999*x34_0/(exp((x87_0)*log(abs(exp((-x84_0 - 13.656822)*log(abs(10.0)))*x83_0))) + 1.0) - 68.422430000000006*x46_0 + 43.20243*x52_0*x53_0 - x88_0 - 178.4239 - (19.734269999999999*x45_0*log(16780.950000000001*x34_0 + 1.0) - 14.509090000000008*x46_0 + 37.886913*x52_0*x53_0 - x88_0 - 307.31920000000002)/(exp((x87_0)*log(abs(exp((-x84_0 - 14.82123)*log(abs(10.0)))*x83_0))) + 1.0))*log(abs(10.0)))
    var x90_0 = 743.05999999999995*x34_0 - 2.4640089999999999*x45_0*x6_0 + 0.19859550000000001*x54_0
    var x91_0 = 2.9375070000000001*x85_0 + 0.23588480000000001*x86_0 + 0.75022860000000002
    var x92_0 = -21.360939999999999*x45_0*log(27535.310000000001*x34_0 + 1.0) + 0.25820969999999999*x57_0
    var x93_0 = exp((-21467.790000000001*x34_0 - 1657.4099999999999*x34_0/(exp((x91_0)*log(abs(exp((-x90_0 - 8.1313220000000008)*log(abs(10.0)))*x83_0))) + 1.0) + 42.707410000000003*x45_0*x6_0 - 2.0273650000000001*x54_0 - x92_0 - 142.7664 - (70.138370000000009*x45_0*x6_0 + 11.28215*x45_0*log(14254.549999999999*x34_0 + 1.0) - 4.7035149999999994*x54_0 - x92_0 - 203.11568)/(exp((x91_0)*log(abs(exp((-x90_0 - 9.3055640000000004)*log(abs(10.0)))*x83_0))) + 1.0))*log(abs(10.0)))
    var x94_0 = vget(X, 2)*vget(X, 8)
    var x95_0 = 5.0000000000000004e-32*x67_0 + 1.5e-32*x81_0
    var x96_0 = ((vget(X, 2))*(vget(X, 2)))*vget(X, 8)
    var x97_0 = x47_0*x75_0
    var x98_0 = 1.0/(exp((1.3*x45_0*(x6_0 - 9.2103403719761836) - 137.42519902360013*x53_0*((0.10857362047581294*x6_0 - 1)*(0.10857362047581294*x6_0 - 1)) - 4.8449999999999998)*log(abs(10.0)))*x83_0 + 1.0)
    var x99_0 = ((vget(X, 8))*(vget(X, 8)))*exp((x98_0)*log(abs(1.1800000000000001e-10*exp(-69500.0*x34_0))))*exp((1.0 - x98_0)*log(abs(8.1250000000000003e-8*x67_0*(1.0 - exp(-6000.0*x34_0))*exp(-52000.0*x34_0))))
    var x100_0 = exp((0.34999999999999998)*log(abs(T)))*x41_0*exp(-102000.0*x34_0)
    var x101_0 = vget(X, 5)*vget(X, 8)*((((
       exp((5.8888600000000002*x46_0 + 7.1969200000000004*x54_0 + 2.2506900000000001*x57_0 - 56.473700000000001 - 2.1690299999999998*x60_0/((((x44_0)*(x44_0)))*(((x44_0)*(x44_0)))) + 0.31788699999999998*x61_0/((x44_0)*(x44_0)*(x44_0)*(x44_0)*(x44_0)))*log(abs(10.0)))
    )) if truthy((T <= 1167.4796423742259)) else ((
       3.1699999999999999e-10*exp(-5207.0*x34_0)
    ))))
    var x102_0 = vget(X, 10)*vget(X, 2)*((((
       5.25e-11*exp(-4430.0*x34_0 + 173900.0/x3_0)
    )) if truthy((T > 200.0)) else ((
       0
    ))))
    var x103_0 = x101_0 - x102_0
    var x104_0 = 6.1739095063118665e-10*exp((0.40999999999999998)*log(abs(T)))
    var x105_0 = x104_0*x21_0
    var x106_0 = x104_0*x38_0
    var x107_0 = x105_0 - x106_0
    var x108_0 = x107_0 + x17_0 - x24_0 + x27_0
    var x109_0 = x80_0*(2.0000000000000002e-31*x67_0 + 6.0000000000000001e-32*x81_0) + x94_0*(-x89_0 - x93_0)
    var x110_0 = x40_0 - x42_0 + x79_0
    var x111_0 = 9.8726896031426014e-7*vget(X, 4)*x68_0
    var x112_0 = -x51_0
    var x113_0 = -x55_0
    var x114_0 = -x37_0 + x49_0
    var x115_0 = x103_0 + x22_0 + x82_0
    var x116_0 = vget(X, 4)*vget(X, 8)
    var x117_0 = x26_0 - x31_0
    var x118_0 = x35_0 - x70_0 + x72_0
    vset(ydot, 0, -x1_0 + x17_0 + x19_0 - x2_0 + x22_0 - x24_0 + x26_0 + x27_0 - x31_0 - x35_0 - x36_0 - x43_0 - x5_0)
    vset(ydot, 1, vget(X, 4)*vget(X, 8)*x55_0 - x49_0 - x50_0 - x51_0 - x64_0 - x78_0)
    vset(ydot, 2, 8.7599999999999997e-10*x100_0 + x103_0 + x108_0 + x109_0 + x43_0 + 2*x5_0 + 2*x50_0 - x58_0 + x63_0 + x78_0 + x79_0 + x80_0*(-6.0000000000000005e-31*x67_0 - 1.8e-31*x81_0) - x82_0 + x94_0*(3*x89_0 + 3*x93_0) - x95_0*x96_0 - x97_0 + 2*x99_0)
    vset(ydot, 3, -x108_0 - x110_0 - x19_0 - x22_0 - x50_0)
    vset(ydot, 4, vget(X, 4)*vget(X, 8)*x113_0 - x1_0 - x111_0 - x112_0 + x74_0 - x76_0 - x97_0)
    vset(ydot, 5, 1.9745379206285203e-6*vget(X, 4)*vget(X, 7)*x67_0 + x1_0 - x107_0 - x114_0 - x115_0 - x2_0 + x69_0 - x77_0)
    vset(ydot, 6, -x5_0 + x64_0 - x79_0)
    vset(ydot, 7, x105_0 - x106_0 - x111_0 + x2_0 - x39_0 - x69_0)
    vset(ydot, 8, -4.3799999999999999e-10*x100_0 - x101_0 + x102_0 + x109_0 + x110_0 + x113_0*x116_0 + x51_0 + x59_0 - x62_0 + x95_0*x96_0 + x96_0*(-2.5000000000000002e-32*x67_0 - 7.5000000000000001e-33*x81_0) - x99_0)
    vset(ydot, 9, x114_0 + x66_0 + x97_0)
    vset(ydot, 10, x112_0 + x115_0 + x116_0*x55_0 + x39_0 + x65_0)
    vset(ydot, 11, x117_0)
    vset(ydot, 12, -x117_0 - x118_0)
    vset(ydot, 13, x118_0)


def rhs_eint(state: BurnState, X: SpeciesTensor, z: Float64) -> Float64:
    var T = state.T
    var x0_0 = 9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9)
    var x1_0 = 1.0/x0_0
    var x2_0 = sqrt(T)
    var x3_0 = vget(X, 1) + vget(X, 12) + vget(X, 4)
    var x4_0 = 1.0/T
    var x5_0 = 2.73*z + 2.73
    var x6_0 = T <= 10
    var x7_0 = vget(X, 0)*vget(X, 12)
    var x8_0 = ((((
       1.0/10.0
    )) if truthy((x6_0)) else ((
       x4_0
    ))))
    var x9_0 = sqrt(T)
    var x10_0 = ((((
       sqrt(10.0)
    )) if truthy((x6_0)) else ((
       x9_0
    ))))
    var x11_0 = 1.0/(0.0031622776601683794*x10_0 + 1.0)
    var x12_0 = vget(X, 0)*x11_0
    var x13_0 = vget(X, 2)*x12_0
    var x14_0 = vget(X, 0)*x10_0*((((
       0.63095734448019325
    )) if truthy((x6_0)) else ((
       exp((-0.20000000000000001)*log(abs(T)))
    ))))/(6.3095734448019361e-5*((((
       5.011872336272722
    )) if truthy((x6_0)) else ((
       exp((0.69999999999999996)*log(abs(T)))
    )))) + 1.0)
    var x15_0 = 1.0/x2_0
    var x16_0 = vget(X, 10) + vget(X, 2) + vget(X, 3) + vget(X, 9)
    var x17_0 = vget(X, 1) + 2.0*vget(X, 6) + 2.0*vget(X, 8) + x16_0
    var x18_0 = 1.0/(1000000.0*x15_0/(x17_0*(1.6000000000000001*vget(X, 2)*exp(-160000.0/((T)*(T))) + 1.3999999999999999*vget(X, 8)*exp(-12000.0/(T + 1200.0)))) + 1.0)
    var x19_0 = x11_0*x7_0
    var x20_0 = ((vget(X, 0))*(vget(X, 0)))*vget(X, 12)*x11_0*((((
       0.67810976749343443
    )) if truthy((x6_0)) else ((
       exp((-0.16869999999999999)*log(abs(T)))
    ))))
    var x21_0 = exp((-0.25)*log(abs(T)))
    var x22_0 = sqrt(Pi)
    var x23_0 = log(T)
    var x24_0 = Log10
    var x25_0 = 1.0/x24_0
    var x26_0 = exp((-2)*log(abs(x24_0)))
    var x27_0 = 1.0/(exp((1.3*x25_0*(x23_0 - 9.2103403719761836) - 137.42519902360013*x26_0*((0.10857362047581294*x23_0 - 1)*(0.10857362047581294*x23_0 - 1)) - 4.8449999999999998)*log(abs(10.0)))*x17_0 + 1.0)
    var x28_0 = ((x23_0)*(x23_0))
    var x29_0 = x26_0*x28_0
    var x30_0 = -4.8909149999999997*x23_0*x25_0 + 0.47490300000000002*x29_0 - 133.82830000000001*x4_0
    var x31_0 = exp(-0.0022727272727272726*T)
    var x32_0 = exp(-0.00054054054054054055*T)
    var x33_0 = -2.0563129999999998*x31_0 + 0.58640729999999996*x32_0 + 0.82274429999999998
    var x34_0 = x23_0*x25_0
    var x35_0 = powi_m3(x24_0)
    var x36_0 = ((x23_0)*(x23_0)*(x23_0))*x35_0
    var x37_0 = -69.700860000000006*x25_0*log(40870.379999999997*x4_0 + 1.0) + 4.6331670000000003*x36_0
    var x38_0 = -2.4640089999999999*x23_0*x25_0 + 0.19859550000000001*x29_0 + 743.05999999999995*x4_0
    var x39_0 = 2.9375070000000001*x31_0 + 0.23588480000000001*x32_0 + 0.75022860000000002
    var x40_0 = -21.360939999999999*x25_0*log(27535.310000000001*x4_0 + 1.0) + 0.25820969999999999*x36_0
    var x41_0 = log(((((
       10000.0
    )) if truthy((T >= 10000.0)) else ((
       T
    )))))
    var x42_0 = exp((-4)*log(abs(x24_0)))
    var x43_0 = ((((x41_0)*(x41_0)))*(((x41_0)*(x41_0))))
    var x44_0 = ((x41_0)*(x41_0)*(x41_0))
    var x45_0 = ((x41_0)*(x41_0))
    var x46_0 = vget(X, 2) <= 0.01
    var x47_0 = log(((((
       10000000000.0
    )) if truthy((((((
       False
    )) if truthy((x46_0)) else ((
       vget(X, 2) >= 10000000000.0
    )))))) else ((
       ((((
          0.01
       )) if truthy((x46_0)) else ((
          vget(X, 2)
       ))))
    )))))
    var x48_0 = ((((x47_0)*(x47_0)))*(((x47_0)*(x47_0))))
    var x49_0 = ((x47_0)*(x47_0)*(x47_0))
    var x50_0 = ((x47_0)*(x47_0))
    var x51_0 = powi_m5(x24_0)
    var x52_0 = exp((-8)*log(abs(x24_0)))
    var x53_0 = powi_m7(x24_0)
    var x54_0 = exp((-6)*log(abs(x24_0)))
    var x55_0 = x0_0 >= 0.5
    var x56_0 = 1.0000420000000001*x25_0
    var x57_0 = x0_0 >= 9.9999999999999998e-13
    var x58_0 = ((((
       exp((x56_0*log(((((
          0.5
       )) if truthy((x55_0)) else ((
          x0_0
       ))))) + 2.1498900000000001)*log(abs(10.0)))
    )) if truthy((x57_0)) else ((
       0.0
    ))))
    var x59_0 = vget(X, 0) + vget(X, 11) + vget(X, 13) + vget(X, 5) + vget(X, 6) + vget(X, 7) + vget(X, 8) + x16_0 + x3_0
    var x60_0 = x59_0 <= 9.9999999999999993e-41
    var x61_0 = x0_0 <= 9.9999999999999993e-41
    var x62_0 = x0_0*x22_0*x9_0
    var x63_0 = exp((2.1498900000000001 - 0.69317629274152892*x25_0)*log(abs(10.0)))*x62_0
    var x64_0 = exp((x56_0*log(x0_0) + 2.1498900000000001)*log(abs(10.0)))*x62_0
    var x65_0 = 0.00013612213614898791*vget(X, 0) + 0.24994102282436673*vget(X, 1) + 0.75007714496081457*vget(X, 10) + 0.99972775572710437*vget(X, 11) + 0.99986387786355213*vget(X, 12) + vget(X, 13) + 0.25007714496081457*vget(X, 2) + 0.25021326709726244*vget(X, 3) + 0.49986387786355219*vget(X, 4) + 0.5*vget(X, 5) + 0.50001816778518127*vget(X, 6) + 0.50013612213644787*vget(X, 7) + 0.50015428992162914*vget(X, 8) + 0.7499410228243667*vget(X, 9)
    var x66_0 = 1.0/abs(x65_0)
    var x67_0 = sqrt(x59_0)
    var x68_0 = exp((-2)*log(abs(x65_0)))
    var x69_0 = sqrt(x59_0*x68_0)
    var x70_0 = 1.2500000000000001e-10*vget(X, 0) + 1.2500000000000001e-10*vget(X, 1) + 1.2500000000000001e-10*vget(X, 10) + 1.2500000000000001e-10*vget(X, 11) + 1.2500000000000001e-10*vget(X, 12) + 1.2500000000000001e-10*vget(X, 13) + 1.2500000000000001e-10*vget(X, 2) + 1.2500000000000001e-10*vget(X, 3) + 1.2500000000000001e-10*vget(X, 4) + 1.2500000000000001e-10*vget(X, 5) + 1.2500000000000001e-10*vget(X, 6) + 1.2500000000000001e-10*vget(X, 7) + 1.2500000000000001e-10*vget(X, 8) + 1.2500000000000001e-10*vget(X, 9) <= 9.9999999999999993e-41
    var x71_0 = 28601.610899577994*exp((-0.45000000000000001)*log(abs(x59_0)))
    var x72_0 = 4.985670872372847e-33*exp((3.7599999999999998)*log(abs(T)))*exp(-2197000.0/((T)*(T)*(T)))/(6.0142468035272636e-8*exp((2.1000000000000001)*log(abs(T))) + 1.0) + 1.6e-18*exp(-11700.0*x4_0) + 6.7e-19*exp(-5860.0*x4_0) + 3.0e-24*exp(-510.0*x4_0)
    var x73_0 = T < 2000.0
    var x74_0 = x23_0 - 6.9077552789821368
    var x75_0 = 0.14476482730108395*x23_0 - 1
    var x76_0 = x26_0*((x75_0)*(x75_0))
    var x77_0 = ((x75_0)*(x75_0)*(x75_0))
    var x78_0 = x35_0*x77_0
    var x79_0 = ((((x75_0)*(x75_0)))*(((x75_0)*(x75_0))))
    var x80_0 = ((x75_0)*(x75_0)*(x75_0)*(x75_0)*(x75_0))
    var x81_0 = x54_0*exp((6)*log(abs(x75_0)))
    var x82_0 = x53_0*((x75_0)*(x75_0)*(x75_0)*(x75_0)*(x75_0)*(x75_0)*(x75_0))
    var x83_0 = exp((8)*log(abs(x75_0)))
    var x84_0 = exp((5.0194035000000001*x25_0*x74_0 + 5627.2167698544854*x42_0*x79_0 + 86051.290034608537*x51_0*x80_0 + 9415777.8988952208*x52_0*x83_0 - 75.100986441619156*x76_0 - 1554.3387057364687*x78_0 - 428804.85473346239*x81_0 - 1662263.0320406025*x82_0 - 20.584225)*log(abs(10.0)))
    var x85_0 = T <= 10000.0
    var x86_0 = 0.00020000000000000001*T
    var x87_0 = x86_0 - 6.0
    var x88_0 = x87_0 >= 300.0
    var x89_0 = ((((
       x72_0
    )) if truthy((x73_0)) else (((((
       x84_0
    )) if truthy((x85_0)) else ((
       5.5313336794064847e-19/(exp(((((
          300.0
       )) if truthy((x88_0)) else ((
          x87_0
       ))))) + 1.0)
    )))))))
    var x90_0 = exp((25.0*x25_0)*log(abs(T)))
    var x91_0 = exp((-200.0 + 20000.0/((10.0 + 2.3538526683701997e+17/x90_0)*(1.6889118802245084e-48*x90_0 + 10.0)))*log(abs(10.0)))
    var x92_0 = x42_0*x79_0
    var x93_0 = x51_0*x80_0
    var x94_0 = exp((2.0943374000000001*x25_0*x74_0 + 144.02112655888752*x35_0*x77_0 - 36.814414747418546*x76_0 - 339.5619991617852*x92_0 - 529.07725573213918*x93_0 - 23.962112000000001)*log(abs(10.0)))*vget(X, 8)*x91_0
    var x95_0 = x25_0*x74_0
    var x96_0 = exp((-38.89917505778142*x76_0 + 95.70878894783884*x78_0 - 377.88183430702219*x92_0 + 3018.4974183098116*x93_0 + 2.1892372*x95_0 - 23.689236999999999)*log(abs(10.0)))*vget(X, 13)
    var x97_0 = T > 10.0
    var x98_0 = x85_0  and  x97_0
    var x99_0 = exp((16.666666666666664*x25_0)*log(abs(T)))
    var x100_0 = exp((-200.0 + 20000.0/((10.0 + 785.77199422741614/x99_0)*(5.0592917094448065e-34*x99_0 + 10.0)))*log(abs(10.0)))
    var x101_0 = 1.002560385050777e-22*vget(X, 13)*x100_0
    var x102_0 = exp((0.73442154540113413*x76_0 - 77.855706084264682*x78_0 - 1161.2797752309887*x92_0 + 5059.6285287169567*x93_0 + 1.5714710999999999*x95_0 - 22.089523)*log(abs(10.0)))*vget(X, 1)
    var x103_0 = 1.1825091393820599e-21*vget(X, 1)*x100_0
    var x104_0 = exp((2774.5177117396752*x76_0 + 16037.924047681272*x78_0 + 45902.322591745004*x92_0 + 60522.293708798054*x93_0 + 37.383713*x95_0 - 16.818342000000001)*log(abs(10.0)))*vget(X, 2)
    var x105_0 = T <= 100.0
    var x106_0 = exp((3.5692468000000002*x25_0*x74_0 - 540.77102118284597*x76_0 - 9179.8864335208946*x78_0 - 48562.751069188118*x92_0 - 66875.646562351845*x93_0 - 24.311209000000002)*log(abs(10.0)))*vget(X, 2)
    var x107_0 = T <= 1000.0
    var x108_0 = exp((-177.55453097873294*x76_0 + 1956.911370108365*x78_0 - 12547.661945180447*x92_0 + 24439.250555499191*x93_0 + 4.6450521*x95_0 - 24.311209000000002)*log(abs(10.0)))*vget(X, 2)
    var x109_0 = T <= 6000.0
    var x110_0 = exp((17.997580222853362*x25_0)*log(abs(T)))
    var x111_0 = 1.8623144679125181e-22*exp((-200.0 + 20000.0/((10.0 + 2973.7534532281375/x110_0)*(1.3368457736780898e-34*x110_0 + 10.0)))*log(abs(10.0)))*vget(X, 2)
    var x112_0 = x52_0*x83_0
    var x113_0 = exp((366063607.58415633*x112_0 + 4616.3011562659685*x76_0 + 113122.17137872758*x78_0 + 87115306.05744876*x81_0 + 273295393.17143697*x82_0 + 1672890.7229183144*x92_0 + 15471651.937466398*x93_0 + 16.815729999999999*x95_0 - 21.928795999999998)*log(abs(10.0)))*vget(X, 0)
    var x114_0 = T <= 500.0
    var x115_0 = T > 100
    var x116_0 = x114_0  and  x115_0
    var x117_0 = exp((-33025002.640084207*x112_0 + 44.525106942242758*x76_0 + 1331.8748828877385*x78_0 + 968783.44101153011*x81_0 + 4831859.3594864924*x82_0 - 10763.919849753534*x92_0 - 138531.11016116844*x93_0 + 1.6802758*x95_0 - 22.921188999999998)*log(abs(10.0)))*vget(X, 0)
    var x118_0 = T > 500.0
    var x119_0 = x91_0*((((
       x113_0
    )) if truthy((x116_0)) else (((((
       x117_0
    )) if truthy((x118_0)) else ((
       0
    ))))))) + x94_0 + ((((
       x102_0
    )) if truthy((x98_0)) else ((
       x103_0
    )))) + ((((
       x96_0
    )) if truthy((x98_0)) else ((
       x101_0
    )))) + ((((
       x104_0
    )) if truthy((x105_0)) else (((((
       x106_0
    )) if truthy((x107_0)) else (((((
       x108_0
    )) if truthy((x109_0)) else ((
       x111_0
    ))))))))))
    var x120_0 = x72_0 >= 1.0e-99
    var x121_0 = x113_0*x91_0
    var x122_0 = x104_0 + x94_0
    var x123_0 = x102_0 + x96_0
    var x124_0 = x122_0 + x123_0
    var x125_0 = x121_0 + x124_0 >= 1.0e-99
    var x126_0 = x123_0 + x94_0
    var x127_0 = x106_0 + x126_0
    var x128_0 = x121_0 + x127_0 >= 1.0e-99
    var x129_0 = x108_0 + x126_0
    var x130_0 = x121_0 + x129_0 >= 1.0e-99
    var x131_0 = x111_0 + x126_0
    var x132_0 = x121_0 + x131_0 >= 1.0e-99
    var x133_0 = x101_0 + x103_0
    var x134_0 = x122_0 + x133_0
    var x135_0 = x121_0 + x134_0 >= 1.0e-99
    var x136_0 = x133_0 + x94_0
    var x137_0 = x106_0 + x136_0
    var x138_0 = x121_0 + x137_0 >= 1.0e-99
    var x139_0 = x108_0 + x136_0
    var x140_0 = x121_0 + x139_0 >= 1.0e-99
    var x141_0 = x111_0 + x136_0
    var x142_0 = x121_0 + x141_0 >= 1.0e-99
    var x143_0 = x117_0*x91_0
    var x144_0 = x124_0 + x143_0 >= 1.0e-99
    var x145_0 = x127_0 + x143_0 >= 1.0e-99
    var x146_0 = x129_0 + x143_0 >= 1.0e-99
    var x147_0 = x131_0 + x143_0 >= 1.0e-99
    var x148_0 = x134_0 + x143_0 >= 1.0e-99
    var x149_0 = x137_0 + x143_0 >= 1.0e-99
    var x150_0 = x139_0 + x143_0 >= 1.0e-99
    var x151_0 = x141_0 + x143_0 >= 1.0e-99
    var x152_0 = x124_0 >= 1.0e-99
    var x153_0 = x127_0 >= 1.0e-99
    var x154_0 = x129_0 >= 1.0e-99
    var x155_0 = x131_0 >= 1.0e-99
    var x156_0 = x134_0 >= 1.0e-99
    var x157_0 = x137_0 >= 1.0e-99
    var x158_0 = x139_0 >= 1.0e-99
    var x159_0 = x141_0 >= 1.0e-99
    var x160_0 = x84_0 >= 1.0e-99
    var x161_0 = 5.5313336794064847e-19/(0.0024787521766663585*exp(x86_0) + 1.0) >= 1.0e-99
    return (x1_0*(-3.1438547368704001e-21*exp((0.34999999999999998)*log(abs(T)))*vget(X, 0)*vget(X, 8)*exp(-102000.0*x4_0) - 0.00022681492*((((T)*(T)))*(((T)*(T))))*x0_0*x58_0*((((
       1.0
    )) if truthy((((((
       4.8339620236294848e-32/((x63_0 + 2.1986273043946046e-56)*(x63_0 + 2.1986273043946046e-56)) >= 1.0
    )) if truthy((x55_0  and  x57_0  and  x60_0  and  x61_0)) else ((
       ((((
          4.8339620236294848e-32/((x64_0 + 2.1986273043946046e-56)*(x64_0 + 2.1986273043946046e-56)) >= 1.0
       )) if truthy((x57_0  and  x60_0  and  x61_0)) else ((
          ((((
             True
          )) if truthy((x60_0  and  x61_0)) else ((
             ((((
                216.48287161311649/((x63_0*x66_0 + 1.471335691176954e-39)*(x63_0*x66_0 + 1.471335691176954e-39)) >= 1.0
             )) if truthy((x55_0  and  x57_0  and  x60_0)) else ((
                ((((
                   216.48287161311649/((x64_0*x66_0 + 1.471335691176954e-39)*(x64_0*x66_0 + 1.471335691176954e-39)) >= 1.0
                )) if truthy((x57_0  and  x60_0)) else ((
                   ((((
                      True
                   )) if truthy((x60_0)) else ((
                      ((((
                         4.833962023629485e-72/((x63_0*x67_0 + 2.1986273043946045e-76)*(x63_0*x67_0 + 2.1986273043946045e-76)) >= 1.0
                      )) if truthy((x55_0  and  x57_0  and  x61_0)) else ((
                         ((((
                            4.833962023629485e-72/((x64_0*x67_0 + 2.1986273043946045e-76)*(x64_0*x67_0 + 2.1986273043946045e-76)) >= 1.0
                         )) if truthy((x57_0  and  x61_0)) else ((
                            ((((
                               True
                            )) if truthy((x61_0)) else ((
                               ((((
                                  2.1648287161311648e-38/((x63_0*x69_0 + 1.471335691176954e-59)*(x63_0*x69_0 + 1.471335691176954e-59)) >= 1.0
                               )) if truthy((x55_0  and  x57_0)) else ((
                                  ((((
                                     2.1648287161311648e-38/((x64_0*x69_0 + 1.471335691176954e-59)*(x64_0*x69_0 + 1.471335691176954e-59)) >= 1.0
                                  )) if truthy((x57_0)) else ((
                                     True
                                  ))))
                               ))))
                            ))))
                         ))))
                      ))))
                   ))))
                ))))
             ))))
          ))))
       ))))
    )))))) else ((
       483396202.36294854/((x58_0*x62_0*sqrt(((((
          9.9999999999999993e-41
       )) if truthy((x60_0)) else ((
          x59_0
       ))))*((((
          1.0e+80
       )) if truthy((x61_0)) else ((
          2.232953576238777e+46*x68_0
       ))))) + 2.1986273043946046e-36)*(x58_0*x62_0*sqrt(((((
          9.9999999999999993e-41
       )) if truthy((x60_0)) else ((
          x59_0
       ))))*((((
          1.0e+80
       )) if truthy((x61_0)) else ((
          2.232953576238777e+46*x68_0
       ))))) + 2.1986273043946046e-36))
    )))) + 0.00084373771595996178*T*(1.3806479999999999e-16*vget(X, 0) + 1.3806479999999999e-16*vget(X, 1) + 1.3806479999999999e-16*vget(X, 10) + 1.3806479999999999e-16*vget(X, 11) + 1.3806479999999999e-16*vget(X, 12) + 1.3806479999999999e-16*vget(X, 13) + 1.3806479999999999e-16*vget(X, 2) + 1.3806479999999999e-16*vget(X, 3) + 1.3806479999999999e-16*vget(X, 4) + 1.3806479999999999e-16*vget(X, 5) + 1.3806479999999999e-16*vget(X, 6) + 1.3806479999999999e-16*vget(X, 7) + 1.3806479999999999e-16*vget(X, 8) + 1.3806479999999999e-16*vget(X, 9))/(sqrt(x1_0)*x22_0) - 2.1299999999999999e-27*vget(X, 0)*x2_0*(4.0*vget(X, 11) + x3_0) - 5.6500000000000001e-36*vget(X, 0)*(T - x5_0)*((((z + 1.0)*(z + 1.0)))*(((z + 1.0)*(z + 1.0)))) - 3.4635323838154264e-26*vget(X, 1)*x14_0 - 1.3854129535261706e-25*vget(X, 11)*x14_0 - 9.3799999999999993e-22*vget(X, 13)*x10_0*x12_0*exp(-285335.40000000002*x8_0) + 7.1777505408000004e-12*((vget(X, 2))*(vget(X, 2))*(vget(X, 2)))*x18_0*(2.0000000000000002e-31*x15_0 + 6.0000000000000001e-32*x21_0) + 7.1777505408000004e-12*((vget(X, 2))*(vget(X, 2)))*vget(X, 8)*x18_0*(2.5000000000000002e-32*x15_0 + 7.5000000000000001e-33*x21_0) + 5.6556829037999995e-12*vget(X, 2)*vget(X, 3)*x18_0*(1.3500000000000001e-9*exp((0.098492999999999997)*log(abs(T))) + 4.4350199999999998e-10*exp((0.55610000000000004)*log(abs(T))) + 3.7408500000000004e-16*exp((2.1825999999999999)*log(abs(T))))/(0.0061910000000000003*exp((1.0461)*log(abs(T))) + 8.9711999999999997e-11*exp((3.0424000000000002)*log(abs(T))) + 3.2575999999999999e-14*exp((3.7740999999999998)*log(abs(T))) + 1.0) + 1.75918975308e-21*vget(X, 2)*vget(X, 6)*x18_0 - 7.1777505408000004e-12*vget(X, 2)*vget(X, 8)*(exp((42.707410000000003*x23_0*x25_0 - 2.0273650000000001*x29_0 - 21467.790000000001*x4_0 - 1657.4099999999999*x4_0/(exp((x39_0)*log(abs(exp((-x38_0 - 8.1313220000000008)*log(abs(10.0)))*x17_0))) + 1.0) - x40_0 - 142.7664 - (70.138370000000009*x23_0*x25_0 + 11.28215*x25_0*log(14254.549999999999*x4_0 + 1.0) - 4.7035149999999994*x29_0 - x40_0 - 203.11568)/(exp((x39_0)*log(abs(exp((-x38_0 - 9.3055640000000004)*log(abs(10.0)))*x17_0))) + 1.0))*log(abs(10.0))) + exp((43.20243*x26_0*x28_0 - 68.422430000000006*x34_0 - x37_0 - 23705.700000000001*x4_0 - 2080.4099999999999*x4_0/(exp((x33_0)*log(abs(exp((-x30_0 - 13.656822)*log(abs(10.0)))*x17_0))) + 1.0) - 178.4239 - (19.734269999999999*x25_0*log(16780.950000000001*x4_0 + 1.0) + 37.886913*x26_0*x28_0 - 14.509090000000008*x34_0 - x37_0 - 307.31920000000002)/(exp((x33_0)*log(abs(exp((-x30_0 - 14.82123)*log(abs(10.0)))*x17_0))) + 1.0))*log(abs(10.0)))) - 7.1777505408000004e-12*((vget(X, 8))*(vget(X, 8)))*exp((x27_0)*log(abs(1.1800000000000001e-10*exp(-69500.0*x4_0))))*exp((1.0 - x27_0)*log(abs(8.1250000000000003e-8*x15_0*(1.0 - exp(-6000.0*x4_0))*exp(-52000.0*x4_0)))) - 1.2700000000000001e-21*x10_0*x13_0*exp(-157809.10000000001*x8_0) - 4.9500000000000001e-22*x10_0*x19_0*exp(-631515.0*x8_0) - 7.4999999999999996e-19*x13_0*exp(-118348.0*x8_0) - 5.5399999999999998e-17*x19_0*((((
       0.4008667176273028
    )) if truthy((x6_0)) else ((
       exp((-0.39700000000000002)*log(abs(T)))
    ))))*exp(-473638.0*x8_0) - 5.0099999999999997e-27*x20_0*exp(-55338.0*x8_0) - 9.1000000000000001e-27*x20_0*exp(-13179.0*x8_0) - 1.24e-13*x7_0*(1.0 + 0.29999999999999999*exp(-94000.0*x8_0))*((((
       0.031622776601683791
    )) if truthy((x6_0)) else ((
       exp((-1.5)*log(abs(T)))
    ))))*exp(-470000.0*x8_0) - 1.5499999999999999e-26*x7_0*((((
       2.3157944032250755
    )) if truthy((x6_0)) else ((
       exp((0.36470000000000002)*log(abs(T)))
    )))) - ((((
       0
    )) if truthy((T < 2.0)) else ((
       vget(X, 8)*((((
          1.0
       )) if truthy((((((
          True
       )) if truthy((x70_0)) else ((
          x71_0 >= 1.0
       )))))) else ((
          ((((
             1.000000000000001e+18
          )) if truthy((x70_0)) else ((
             x71_0
          ))))
       ))))*((((
          x119_0*x89_0/(x119_0 + x89_0)
       )) if truthy((((((
          x120_0  and  x125_0
       )) if truthy((x105_0  and  x114_0  and  x115_0  and  x73_0  and  x85_0  and  x97_0)) else ((
          ((((
             x120_0  and  x128_0
          )) if truthy((x107_0  and  x114_0  and  x115_0  and  x73_0  and  x85_0  and  x97_0)) else ((
             ((((
                x120_0  and  x130_0
             )) if truthy((x109_0  and  x114_0  and  x115_0  and  x73_0  and  x85_0  and  x97_0)) else ((
                ((((
                   x120_0  and  x132_0
                )) if truthy((x114_0  and  x115_0  and  x73_0  and  x85_0  and  x97_0)) else ((
                   ((((
                      x120_0  and  x135_0
                   )) if truthy((x105_0  and  x114_0  and  x115_0  and  x73_0)) else ((
                      ((((
                         x120_0  and  x138_0
                      )) if truthy((x107_0  and  x114_0  and  x115_0  and  x73_0)) else ((
                         ((((
                            x120_0  and  x140_0
                         )) if truthy((x109_0  and  x114_0  and  x115_0  and  x73_0)) else ((
                            ((((
                               x120_0  and  x142_0
                            )) if truthy((x114_0  and  x115_0  and  x73_0)) else ((
                               ((((
                                  x120_0  and  x144_0
                               )) if truthy((x105_0  and  x118_0  and  x73_0  and  x85_0  and  x97_0)) else ((
                                  ((((
                                     x120_0  and  x145_0
                                  )) if truthy((x107_0  and  x118_0  and  x73_0  and  x85_0  and  x97_0)) else ((
                                     ((((
                                        x120_0  and  x146_0
                                     )) if truthy((x109_0  and  x118_0  and  x73_0  and  x85_0  and  x97_0)) else ((
                                        ((((
                                           x120_0  and  x147_0
                                        )) if truthy((x118_0  and  x73_0  and  x85_0  and  x97_0)) else ((
                                           ((((
                                              x120_0  and  x148_0
                                           )) if truthy((x105_0  and  x118_0  and  x73_0)) else ((
                                              ((((
                                                 x120_0  and  x149_0
                                              )) if truthy((x107_0  and  x118_0  and  x73_0)) else ((
                                                 ((((
                                                    x120_0  and  x150_0
                                                 )) if truthy((x109_0  and  x118_0  and  x73_0)) else ((
                                                    ((((
                                                       x120_0  and  x151_0
                                                    )) if truthy((x118_0  and  x73_0)) else ((
                                                       ((((
                                                          x120_0  and  x152_0
                                                       )) if truthy((x105_0  and  x73_0  and  x85_0  and  x97_0)) else ((
                                                          ((((
                                                             x120_0  and  x153_0
                                                          )) if truthy((x107_0  and  x73_0  and  x85_0  and  x97_0)) else ((
                                                             ((((
                                                                x120_0  and  x154_0
                                                             )) if truthy((x109_0  and  x73_0  and  x85_0  and  x97_0)) else ((
                                                                ((((
                                                                   x120_0  and  x155_0
                                                                )) if truthy((x73_0  and  x85_0  and  x97_0)) else ((
                                                                   ((((
                                                                      x120_0  and  x156_0
                                                                   )) if truthy((x105_0  and  x73_0)) else ((
                                                                      ((((
                                                                         x120_0  and  x157_0
                                                                      )) if truthy((x107_0  and  x73_0)) else ((
                                                                         ((((
                                                                            x120_0  and  x158_0
                                                                         )) if truthy((x109_0  and  x73_0)) else ((
                                                                            ((((
                                                                               x120_0  and  x159_0
                                                                            )) if truthy((x73_0)) else ((
                                                                               ((((
                                                                                  x125_0  and  x160_0
                                                                               )) if truthy((x105_0  and  x114_0  and  x115_0  and  x85_0  and  x97_0)) else ((
                                                                                  ((((
                                                                                     x128_0  and  x160_0
                                                                                  )) if truthy((x107_0  and  x114_0  and  x115_0  and  x85_0  and  x97_0)) else ((
                                                                                     ((((
                                                                                        x130_0  and  x160_0
                                                                                     )) if truthy((x109_0  and  x114_0  and  x115_0  and  x85_0  and  x97_0)) else ((
                                                                                        ((((
                                                                                           x132_0  and  x160_0
                                                                                        )) if truthy((x114_0  and  x115_0  and  x85_0  and  x97_0)) else ((
                                                                                           ((((
                                                                                              x135_0  and  x160_0
                                                                                           )) if truthy((x105_0  and  x114_0  and  x115_0  and  x85_0)) else ((
                                                                                              ((((
                                                                                                 x138_0  and  x160_0
                                                                                              )) if truthy((x107_0  and  x114_0  and  x115_0  and  x85_0)) else ((
                                                                                                 ((((
                                                                                                    x140_0  and  x160_0
                                                                                                 )) if truthy((x109_0  and  x114_0  and  x115_0  and  x85_0)) else ((
                                                                                                    ((((
                                                                                                       x142_0  and  x160_0
                                                                                                    )) if truthy((x114_0  and  x115_0  and  x85_0)) else ((
                                                                                                       ((((
                                                                                                          x144_0  and  x160_0
                                                                                                       )) if truthy((x105_0  and  x118_0  and  x85_0  and  x97_0)) else ((
                                                                                                          ((((
                                                                                                             x145_0  and  x160_0
                                                                                                          )) if truthy((x107_0  and  x118_0  and  x85_0  and  x97_0)) else ((
                                                                                                             ((((
                                                                                                                x146_0  and  x160_0
                                                                                                             )) if truthy((x109_0  and  x118_0  and  x85_0  and  x97_0)) else ((
                                                                                                                ((((
                                                                                                                   x147_0  and  x160_0
                                                                                                                )) if truthy((x118_0  and  x85_0  and  x97_0)) else ((
                                                                                                                   ((((
                                                                                                                      x148_0  and  x160_0
                                                                                                                   )) if truthy((x105_0  and  x118_0  and  x85_0)) else ((
                                                                                                                      ((((
                                                                                                                         x149_0  and  x160_0
                                                                                                                      )) if truthy((x107_0  and  x118_0  and  x85_0)) else ((
                                                                                                                         ((((
                                                                                                                            x150_0  and  x160_0
                                                                                                                         )) if truthy((x109_0  and  x118_0  and  x85_0)) else ((
                                                                                                                            ((((
                                                                                                                               x151_0  and  x160_0
                                                                                                                            )) if truthy((x118_0  and  x85_0)) else ((
                                                                                                                               ((((
                                                                                                                                  x152_0  and  x160_0
                                                                                                                               )) if truthy((x105_0  and  x85_0  and  x97_0)) else ((
                                                                                                                                  ((((
                                                                                                                                     x153_0  and  x160_0
                                                                                                                                  )) if truthy((x107_0  and  x85_0  and  x97_0)) else ((
                                                                                                                                     ((((
                                                                                                                                        x154_0  and  x160_0
                                                                                                                                     )) if truthy((x109_0  and  x85_0  and  x97_0)) else ((
                                                                                                                                        ((((
                                                                                                                                           x155_0  and  x160_0
                                                                                                                                        )) if truthy((x98_0)) else ((
                                                                                                                                           ((((
                                                                                                                                              x156_0  and  x160_0
                                                                                                                                           )) if truthy((x105_0  and  x85_0)) else ((
                                                                                                                                              ((((
                                                                                                                                                 x157_0  and  x160_0
                                                                                                                                              )) if truthy((x107_0  and  x85_0)) else ((
                                                                                                                                                 ((((
                                                                                                                                                    x158_0  and  x160_0
                                                                                                                                                 )) if truthy((x109_0  and  x85_0)) else ((
                                                                                                                                                    ((((
                                                                                                                                                       x159_0  and  x160_0
                                                                                                                                                    )) if truthy((x85_0)) else ((
                                                                                                                                                       ((((
                                                                                                                                                          False
                                                                                                                                                       )) if truthy((x88_0  and  (x105_0  or  x88_0)  and  (x107_0  or  x88_0)  and  (x109_0  or  x88_0)  and  (x114_0  or  x88_0)  and  (x115_0  or  x88_0)  and  (x118_0  or  x88_0)  and  (x105_0  or  x107_0  or  x88_0)  and  (x105_0  or  x109_0  or  x88_0)  and  (x105_0  or  x114_0  or  x88_0)  and  (x105_0  or  x115_0  or  x88_0)  and  (x105_0  or  x118_0  or  x88_0)  and  (x107_0  or  x109_0  or  x88_0)  and  (x107_0  or  x114_0  or  x88_0)  and  (x107_0  or  x115_0  or  x88_0)  and  (x107_0  or  x118_0  or  x88_0)  and  (x109_0  or  x114_0  or  x88_0)  and  (x109_0  or  x115_0  or  x88_0)  and  (x109_0  or  x118_0  or  x88_0)  and  (x114_0  or  x115_0  or  x88_0)  and  (x115_0  or  x118_0  or  x88_0)  and  (x105_0  or  x107_0  or  x109_0  or  x88_0)  and  (x105_0  or  x107_0  or  x114_0  or  x88_0)  and  (x105_0  or  x107_0  or  x115_0  or  x88_0)  and  (x105_0  or  x107_0  or  x118_0  or  x88_0)  and  (x105_0  or  x109_0  or  x114_0  or  x88_0)  and  (x105_0  or  x109_0  or  x115_0  or  x88_0)  and  (x105_0  or  x109_0  or  x118_0  or  x88_0)  and  (x105_0  or  x114_0  or  x115_0  or  x88_0)  and  (x105_0  or  x115_0  or  x118_0  or  x88_0)  and  (x107_0  or  x109_0  or  x114_0  or  x88_0)  and  (x107_0  or  x109_0  or  x115_0  or  x88_0)  and  (x107_0  or  x109_0  or  x118_0  or  x88_0)  and  (x107_0  or  x114_0  or  x115_0  or  x88_0)  and  (x107_0  or  x115_0  or  x118_0  or  x88_0)  and  (x109_0  or  x114_0  or  x115_0  or  x88_0)  and  (x109_0  or  x115_0  or  x118_0  or  x88_0)  and  (x105_0  or  x107_0  or  x109_0  or  x114_0  or  x88_0)  and  (x105_0  or  x107_0  or  x109_0  or  x115_0  or  x88_0)  and  (x105_0  or  x107_0  or  x109_0  or  x118_0  or  x88_0)  and  (x105_0  or  x107_0  or  x114_0  or  x115_0  or  x88_0)  and  (x105_0  or  x107_0  or  x115_0  or  x118_0  or  x88_0)  and  (x105_0  or  x109_0  or  x114_0  or  x115_0  or  x88_0)  and  (x105_0  or  x109_0  or  x115_0  or  x118_0  or  x88_0)  and  (x107_0  or  x109_0  or  x114_0  or  x115_0  or  x88_0)  and  (x107_0  or  x109_0  or  x115_0  or  x118_0  or  x88_0)  and  (x105_0  or  x107_0  or  x109_0  or  x114_0  or  x115_0  or  x88_0)  and  (x105_0  or  x107_0  or  x109_0  or  x115_0  or  x118_0  or  x88_0))) else ((
                                                                                                                                                          ((((
                                                                                                                                                             x135_0  and  x161_0
                                                                                                                                                          )) if truthy((x105_0  and  x114_0  and  x115_0)) else ((
                                                                                                                                                             ((((
                                                                                                                                                                x138_0  and  x161_0
                                                                                                                                                             )) if truthy((x107_0  and  x114_0  and  x115_0)) else ((
                                                                                                                                                                ((((
                                                                                                                                                                   x140_0  and  x161_0
                                                                                                                                                                )) if truthy((x109_0  and  x114_0  and  x115_0)) else ((
                                                                                                                                                                   ((((
                                                                                                                                                                      x142_0  and  x161_0
                                                                                                                                                                   )) if truthy((x116_0)) else ((
                                                                                                                                                                      ((((
                                                                                                                                                                         x148_0  and  x161_0
                                                                                                                                                                      )) if truthy((x105_0  and  x118_0)) else ((
                                                                                                                                                                         ((((
                                                                                                                                                                            x149_0  and  x161_0
                                                                                                                                                                         )) if truthy((x107_0  and  x118_0)) else ((
                                                                                                                                                                            ((((
                                                                                                                                                                               x150_0  and  x161_0
                                                                                                                                                                            )) if truthy((x109_0  and  x118_0)) else ((
                                                                                                                                                                               x151_0  and  x161_0
                                                                                                                                                                            ))))
                                                                                                                                                                         ))))
                                                                                                                                                                      ))))
                                                                                                                                                                   ))))
                                                                                                                                                                ))))
                                                                                                                                                             ))))
                                                                                                                                                          ))))
                                                                                                                                                       ))))
                                                                                                                                                    ))))
                                                                                                                                                 ))))
                                                                                                                                              ))))
                                                                                                                                           ))))
                                                                                                                                        ))))
                                                                                                                                     ))))
                                                                                                                                  ))))
                                                                                                                               ))))
                                                                                                                            ))))
                                                                                                                         ))))
                                                                                                                      ))))
                                                                                                                   ))))
                                                                                                                ))))
                                                                                                             ))))
                                                                                                          ))))
                                                                                                       ))))
                                                                                                    ))))
                                                                                                 ))))
                                                                                              ))))
                                                                                           ))))
                                                                                        ))))
                                                                                     ))))
                                                                                  ))))
                                                                               ))))
                                                                            ))))
                                                                         ))))
                                                                      ))))
                                                                   ))))
                                                                ))))
                                                             ))))
                                                          ))))
                                                       ))))
                                                    ))))
                                                 ))))
                                              ))))
                                           ))))
                                        ))))
                                     ))))
                                  ))))
                               ))))
                            ))))
                         ))))
                      ))))
                   ))))
                ))))
             ))))
          ))))
       )))))) else ((
          0
       ))))
    )))) - ((((
       exp((21.93385*x25_0*x41_0 + 0.92432999999999998*x25_0*x47_0 + 0.77951999999999999*x26_0*x41_0*x47_0 - 10.19097*x26_0*x45_0 + 0.54962*x26_0*x50_0 - 1.06447*x35_0*x41_0*x50_0 + 2.1990599999999998*x35_0*x44_0 - 0.54262999999999995*x35_0*x45_0*x47_0 - 0.076759999999999995*x35_0*x49_0 + 0.11864*x41_0*x42_0*x49_0 - 0.0036600000000000001*x41_0*x48_0*x51_0 - 0.17333999999999999*x42_0*x43_0 + 0.11711000000000001*x42_0*x44_0*x47_0 + 0.62343000000000004*x42_0*x45_0*x50_0 + 0.0027499999999999998*x42_0*x48_0 - 0.0083499999999999998*x43_0*x47_0*x51_0 + 6.1920000000000003e-5*x43_0*x48_0*x52_0 - 0.001482*x43_0*x49_0*x53_0 + 0.0106*x43_0*x50_0*x54_0 - 0.00066631000000000004*x44_0*x48_0*x53_0 + 0.017590000000000001*x44_0*x49_0*x54_0 - 0.13768*x44_0*x50_0*x51_0 + 0.0025140000000000002*x45_0*x48_0*x54_0 - 0.073660000000000003*x45_0*x49_0*x51_0 - 42.567880000000002)*log(abs(10.0)))*vget(X, 10)
    )) if truthy((T >= x5_0)) else ((
       0
    ))))))


def actual_rhs(state: BurnState, mut ydot: VecTensor):
    var z = redshift()
    var X = SpeciesTensor.stack_allocation()
    for i in range(NumSpec):
        vset(X, i, state.xn[i])
    rhs_specie(state, ydot, X, z)
    var edot = rhs_eint(state, X, z)
    vset(ydot, NetIenuc, edot)


def jac_nuc(state: BurnState, mut jac: MatTensor, X: SpeciesTensor, z: Float64):
    var T = state.T
    var x0_0 = 2.5950363272655348e-10*exp((-0.75)*log(abs(T)))
    var x1_0 = sqrt(T)
    var x2_0 = 1.0/x1_0
    var x3_0 = 7.1999999999999996e-8*x2_0
    var x4_0 = exp(-6.1728395061728397e-5*T)
    var x5_0 = vget(X, 2)*x4_0
    var x6_0 = exp((0.92800000000000005)*log(abs(T)))
    var x7_0 = 1.4000000000000001e-18*x6_0
    var x8_0 = exp(-0.00010729613733905579*T)
    var x9_0 = vget(X, 5)*x8_0
    var x10_0 = exp((0.94999999999999996)*log(abs(T)))
    var x11_0 = 1.3300135414628029e-18*x10_0
    var x12_0 = 1.0/T
    var x13_0 = exp(-46707.0*x12_0)
    var x14_0 = vget(X, 8)*x13_0
    var x15_0 = 35.5*exp((-2.2799999999999998)*log(abs(T)))
    var x16_0 = 0.00060040841663220993*x1_0 + 1.0
    var x17_0 = exp((-1.7524)*log(abs(x16_0)))
    var x18_0 = 0.32668576019240059*x1_0 + 1.0
    var x19_0 = exp((-0.24759999999999999)*log(abs(x18_0)))
    var x20_0 = vget(X, 11)*x19_0
    var x21_0 = x17_0*x20_0
    var x22_0 = 5.7884371785482823e-10*x2_0
    var x23_0 = ((T)*(T))
    var x24_0 = ((T)*(T)*(T))
    var x25_0 = ((((T)*(T)))*(((T)*(T))))
    var x26_0 = T <= 10000.0
    var x27_0 = ((((
       -5.5279999999999998e-28*((T)*(T)*(T)*(T)*(T)) - 2.3088e-11*T + 7.3427999999999993e-15*x23_0 - 7.5474000000000004e-19*x24_0 + 3.3467999999999999e-23*x25_0 + 4.2277999999999996e-8
    )) if truthy((x26_0)) else ((
       0
    ))))
    var x28_0 = 1.4981088130721367e-10*exp((-0.63529999999999998)*log(abs(T)))
    var x29_0 = 8.6173430000000006e-5*T
    var x30_0 = x29_0 <= 9280.0
    var x31_0 = (1.5400000000000001e-9 + 4.6200000000000001e-10*exp(-93988.701501924661*x12_0))*exp(-469943.50750964211*x12_0)
    var x32_0 = ((((
       x28_0
    )) if truthy((x30_0)) else ((
       1250086.112245841*exp((-1.5)*log(abs(T)))*x31_0 + x28_0
    ))))
    var x33_0 = exp((2.360852208681)*log(abs(T)))
    var x34_0 = log(x29_0)
    var x35_0 = ((x34_0)*(x34_0))
    var x36_0 = ((x34_0)*(x34_0)*(x34_0))
    var x37_0 = ((((x34_0)*(x34_0)))*(((x34_0)*(x34_0))))
    var x38_0 = ((x34_0)*(x34_0)*(x34_0)*(x34_0)*(x34_0))
    var x39_0 = exp((6)*log(abs(x34_0)))
    var x40_0 = ((x34_0)*(x34_0)*(x34_0)*(x34_0)*(x34_0)*(x34_0)*(x34_0))
    var x41_0 = exp((8)*log(abs(x34_0)))
    var x42_0 = exp(-0.28274430617039997*x35_0 + 0.01623316639567*x36_0 - 0.033650120313629989*x37_0 + 0.01178329782711*x38_0 - 0.001656194699504*x39_0 + 0.0001068275202678*x40_0 - 2.6312858092069998e-6*x41_0)
    var x43_0 = exp((13.536555999999999)*log(abs(T)))
    var x44_0 = exp(-5.7393287500000003*x35_0 + 1.56315498*x36_0 - 0.28770560000000001*x37_0 + 0.034825597700000002*x38_0 - 0.00263197617*x39_0 + 0.000111954395*x40_0 - 2.0391498499999999e-6*x41_0)
    var x45_0 = exp((23.915965629999999)*log(abs(T)))
    var x46_0 = exp(-10.753230200000001*x35_0 + 3.0580387500000001*x36_0 - 0.56851189000000002*x37_0 + 0.067953912300000002*x38_0 - 0.0050090561*x39_0 + 0.000206723616*x40_0 - 3.6491614100000001e-6*x41_0)
    var x47_0 = exp((43.933476326349997)*log(abs(T)))
    var x48_0 = exp(-18.480669935680002*x35_0 + 4.7016264867590021*x36_0 - 0.76924663344919997*x37_0 + 0.081130420973029999*x38_0 - 0.005324020628287001*x39_0 + 0.00019757053122209999*x40_0 - 3.1655810656650001e-6*x41_0)
    var x49_0 = x29_0 <= 5500.0
    var x50_0 = exp((-0.72411256578268512)*log(abs(T)))
    var x51_0 = ((x34_0)*(x34_0)*(x34_0)*(x34_0)*(x34_0)*(x34_0)*(x34_0)*(x34_0)*(x34_0))
    var x52_0 = exp(-0.02026044731984691*x35_0 - 0.002380861877349834*x36_0 - 0.00032126052131887958*x37_0 - 1.421502914054107e-5*x38_0 + 4.9891089202995129e-6*x39_0 + 5.7556141375757583e-7*x40_0 - 1.8567670397752609e-8*x41_0 - 3.0711352431965949e-9*x51_0)
    var x53_0 = ((((
       x28_0
    )) if truthy((x49_0)) else ((
       3.2867337024382687e-10*x50_0*x52_0
    ))))
    var x54_0 = 1.0e-8*exp((-0.40000000000000002)*log(abs(T)))
    var x55_0 = 2.6534040307116387e-9*exp((-0.10000000000000001)*log(abs(T)))
    var x56_0 = 0.0061910000000000003*exp((1.0461)*log(abs(T))) + 8.9711999999999997e-11*exp((3.0424000000000002)*log(abs(T))) + 3.2575999999999999e-14*exp((3.7740999999999998)*log(abs(T))) + 1.0
    var x57_0 = 1.0/x56_0
    var x58_0 = 1.3500000000000001e-9*exp((0.098492999999999997)*log(abs(T))) + 4.4350199999999998e-10*exp((0.55610000000000004)*log(abs(T))) + 3.7408500000000004e-16*exp((2.1825999999999999)*log(abs(T)))
    var x59_0 = x57_0*x58_0
    var x60_0 = 5.9082438637265071e-70*x43_0
    var x61_0 = T <= 1160.0
    var x62_0 = exp(-0.14210135215541481*x35_0 + 0.0084644553866299998*x36_0 - 0.0014327641212992001*x37_0 + 0.00020122502847909999*x38_0 + 8.6639632430900003e-5*x39_0 - 2.5850096802639999e-5*x40_0 + 2.4555011970391999e-6*x41_0 - 8.0683824611800006e-8*x51_0)
    var x63_0 = 3.3178155742407614e-14*exp((1.1394493358416311)*log(abs(T)))*x62_0
    var x64_0 = ((((
       1.4643482606109061e-16*exp((1.78186)*log(abs(T)))
    )) if truthy((x61_0)) else ((
       x63_0
    ))))
    var x65_0 = 3.7903999274394518e-18*x33_0
    var x66_0 = vget(X, 0)*x17_0
    var x67_0 = 3.8571873359681582e-209*x47_0*x48_0
    var x68_0 = 4.3524079114767552e-117*x45_0
    var x69_0 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x70_0 = 2.0860422997526066e-16*x69_0
    var x71_0 = 3.4767371836380304e-16*x69_0
    var x72_0 = 2.6534040307116389e-10*exp((-1.1000000000000001)*log(abs(T)))
    var x73_0 = vget(X, 0)/exp((3.0/2.0)*log(abs(T)))
    var x74_0 = vget(X, 0)*x5_0
    var x75_0 = vget(X, 0)*x9_0
    var x76_0 = vget(X, 0)*x14_0
    var x77_0 = vget(X, 2)*vget(X, 3)
    var x78_0 = vget(X, 3)*x42_0
    var x79_0 = vget(X, 2)*x44_0
    var x80_0 = vget(X, 13)*x46_0
    var x81_0 = vget(X, 0)*vget(X, 12)
    var x82_0 = -9.5174852894472843e-11*exp((-1.6353)*log(abs(T)))
    var x83_0 = exp((-3.5)*log(abs(T)))
    var x84_0 = x12_0*x34_0
    var x85_0 = x12_0*x36_0
    var x86_0 = x12_0*x38_0
    var x87_0 = x12_0*x40_0
    var x88_0 = x12_0*x41_0
    mset(jac, 0, 0, -vget(X, 1)*x53_0 - vget(X, 12)*x32_0 + 3.8571873359681582e-209*vget(X, 12)*x47_0*x48_0 + 4.3524079114767552e-117*vget(X, 13)*x45_0*x46_0 + 5.9082438637265071e-70*vget(X, 2)*x43_0*x44_0 + 3.7903999274394518e-18*vget(X, 3)*x33_0*x42_0 - vget(X, 4)*x0_0 - vget(X, 6)*x27_0 - vget(X, 9)*x3_0 - x11_0*x9_0 - x14_0*x15_0 - x21_0*x22_0 - x5_0*x7_0)
    mset(jac, 0, 1, -vget(X, 0)*x53_0 + vget(X, 3)*x54_0)
    mset(jac, 0, 2, -vget(X, 0)*x4_0*x7_0 + vget(X, 0)*x44_0*x60_0 + vget(X, 3)*x59_0 + vget(X, 3)*x64_0 + vget(X, 7)*x55_0)
    mset(jac, 0, 3, vget(X, 0)*x42_0*x65_0 + vget(X, 1)*x54_0 + vget(X, 2)*x59_0 + vget(X, 2)*x64_0 + vget(X, 5)*x55_0)
    mset(jac, 0, 4, -vget(X, 0)*x0_0)
    mset(jac, 0, 5, -vget(X, 0)*x11_0*x8_0 + vget(X, 3)*x55_0)
    mset(jac, 0, 6, -vget(X, 0)*x27_0)
    mset(jac, 0, 7, vget(X, 2)*x55_0)
    mset(jac, 0, 8, -vget(X, 0)*x13_0*x15_0)
    mset(jac, 0, 9, -vget(X, 0)*x3_0)
    mset(jac, 0, 10, 0)
    mset(jac, 0, 11, -x19_0*x22_0*x66_0)
    mset(jac, 0, 12, -vget(X, 0)*x32_0 + vget(X, 0)*x67_0)
    mset(jac, 0, 13, vget(X, 0)*x46_0*x68_0)
    mset(jac, 0, 14, (-1658098.5*exp((-4.2799999999999994)*log(abs(T)))*x76_0 + 80.939999999999998*exp((-3.2799999999999998)*log(abs(T)))*x76_0 + 1.9462772454491511e-10*exp((-1.75)*log(abs(T)))*vget(X, 0)*vget(X, 4) - 4.0000000000000002e-9*exp((-1.3999999999999999)*log(abs(T)))*vget(X, 1)*vget(X, 3) - 1.2992000000000002e-18*exp((-0.071999999999999953)*log(abs(T)))*x74_0 - 1.2635128643896626e-18*exp((-0.050000000000000044)*log(abs(T)))*x75_0 + 8.9485740404797324e-18*exp((1.360852208681)*log(abs(T)))*vget(X, 0)*x78_0 + 7.997727392299023e-69*exp((12.536555999999999)*log(abs(T)))*vget(X, 0)*x79_0 + 1.0409203801861816e-115*exp((22.915965629999999)*log(abs(T)))*vget(X, 0)*x80_0 + 1.694596485110541e-207*exp((42.933476326349997)*log(abs(T)))*x48_0*x81_0 - vget(X, 0)*vget(X, 1)*((((
       x82_0
    )) if truthy((x49_0)) else ((
       -2.3799651743169991e-10*exp((-1.724112565782685)*log(abs(T)))*x52_0 + 3.2867337024382687e-10*x50_0*x52_0*(-0.0071425856320495021*x12_0*x35_0 - 7.1075145702705346e-5*x12_0*x37_0 + 2.9934653521797078e-5*x12_0*x38_0 + 4.0289298963030308e-6*x12_0*x39_0 - 0.04052089463969382*x84_0 - 0.0012850420852755183*x85_0 - 1.4854136318202087e-7*x87_0 - 2.7640217188769353e-8*x88_0)
    )))) - vget(X, 0)*vget(X, 6)*((((
       1.4685599999999999e-14*T - 2.2642200000000001e-18*x23_0 + 1.3387199999999999e-22*x24_0 - 2.7639999999999999e-27*x25_0 - 2.3088e-11
    )) if truthy((x26_0)) else ((
       0
    )))) + 3.0451686126851684e-13*vget(X, 0)*x12_0*exp((-2.7523999999999997)*log(abs(x16_0)))*x20_0 + vget(X, 0)*x60_0*x79_0*(4.6894649399999997*x12_0*x35_0 + 0.1741279885*x12_0*x37_0 + 0.00078368076500000001*x12_0*x39_0 - 11.478657500000001*x84_0 - 1.1508224*x85_0 - 0.015791857020000001*x86_0 - 1.6313198799999999e-5*x87_0) + vget(X, 0)*x65_0*x78_0*(0.048699499187009998*x12_0*x35_0 + 0.058916489135550004*x12_0*x37_0 + 0.00074779264187460007*x12_0*x39_0 - 0.56548861234079995*x84_0 - 0.13460048125451995*x85_0 - 0.009937168197024001*x86_0 - 2.1050286473655998e-5*x87_0) + vget(X, 0)*x68_0*x80_0*(9.1741162500000009*x12_0*x35_0 + 0.33976956150000004*x12_0*x37_0 + 0.001447065312*x12_0*x39_0 - 21.506460400000002*x84_0 - 2.2740475600000001*x85_0 - 0.030054336600000002*x86_0 - 2.9193291280000001e-5*x87_0) + 2.3410580000000002e-11*vget(X, 11)*x12_0*exp((-1.2476)*log(abs(x18_0)))*x66_0 - vget(X, 2)*vget(X, 7)*x72_0 - vget(X, 3)*vget(X, 5)*x72_0 + 3.5999999999999998e-8*vget(X, 9)*x73_0 + 1.4270531560759686e-22*x10_0*x75_0 + 2.8942185892741411e-10*x21_0*x73_0 + x57_0*x77_0*(1.3296555000000001e-10*exp((-0.90150700000000006)*log(abs(T))) + 2.466314622e-10*exp((-0.44389999999999996)*log(abs(T))) + 8.1647792100000001e-16*exp((1.1825999999999999)*log(abs(T)))) + 8.6419753086419757e-23*x6_0*x74_0 + x67_0*x81_0*(14.104879460277006*x12_0*x35_0 + 0.40565210486515002*x12_0*x37_0 + 0.0013829937185547*x12_0*x39_0 - 36.961339871360003*x84_0 - 3.0769865337967999*x85_0 - 0.031944123769722006*x86_0 - 2.5324648525320001e-5*x87_0) + x77_0*((((
       2.6092635916521491e-16*exp((0.78186)*log(abs(T)))
    )) if truthy((x61_0)) else ((
       3.7804827525136553e-14*exp((0.13944933584163111)*log(abs(T)))*x62_0 + x63_0*(0.025393366159889998*x12_0*x35_0 + 0.0010061251423955*x12_0*x37_0 + 0.00051983779458540007*x12_0*x38_0 - 0.00018095067761848*x12_0*x39_0 + 1.9644009576313599e-5*x12_0*x40_0 - 0.28420270431082961*x84_0 - 0.0057310564851968003*x85_0 - 7.2615442150620009e-7*x88_0)
    )))) - x81_0*((((
       x82_0
    )) if truthy((x30_0)) else ((
       -1875129.1683687614*exp((-2.5)*log(abs(T)))*x31_0 + 587469852277.90271*x31_0*x83_0 + x82_0 + 54.282214350476039*x83_0*exp(-563932.20901156683*x12_0)
    )))) + x58_0*x77_0*(-0.0064764051000000007*exp((0.04610000000000003)*log(abs(T))) - 2.7293978880000002e-10*exp((2.0424000000000002)*log(abs(T))) - 1.229450816e-13*exp((2.7740999999999998)*log(abs(T))))/((x56_0)*(x56_0)))/(vget(X, 0)*x70_0 + vget(X, 1)*x70_0 + vget(X, 10)*x71_0 + vget(X, 11)*x70_0 + vget(X, 12)*x70_0 + vget(X, 13)*x70_0 + vget(X, 2)*x70_0 + vget(X, 3)*x70_0 + vget(X, 4)*x70_0 + vget(X, 5)*x70_0 + vget(X, 6)*x71_0 + vget(X, 7)*x70_0 + vget(X, 8)*x71_0 + vget(X, 9)*x71_0))
    var x0_1 = 8.6173430000000006e-5*T
    var x1_1 = log(x0_1)
    var x2_1 = ((x1_1)*(x1_1))
    var x3_1 = ((x1_1)*(x1_1)*(x1_1))
    var x4_1 = ((((x1_1)*(x1_1)))*(((x1_1)*(x1_1))))
    var x5_1 = ((x1_1)*(x1_1)*(x1_1)*(x1_1)*(x1_1))
    var x6_1 = exp((6)*log(abs(x1_1)))
    var x7_1 = ((x1_1)*(x1_1)*(x1_1)*(x1_1)*(x1_1)*(x1_1)*(x1_1))
    var x8_1 = exp((8)*log(abs(x1_1)))
    var x9_1 = exp(-5.7393287500000003*x2_1 + 1.56315498*x3_1 - 0.28770560000000001*x4_1 + 0.034825597700000002*x5_1 - 0.00263197617*x6_1 + 0.000111954395*x7_1 - 2.0391498499999999e-6*x8_1)
    var x10_1 = exp((13.536555999999999)*log(abs(T)))
    var x11_1 = 5.9082438637265071e-70*x10_1
    var x12_1 = x0_1 <= 5500.0
    var x13_1 = exp((-0.72411256578268512)*log(abs(T)))
    var x14_1 = exp(-3.0711352431965949e-9*((x1_1)*(x1_1)*(x1_1)*(x1_1)*(x1_1)*(x1_1)*(x1_1)*(x1_1)*(x1_1)) - 0.02026044731984691*x2_1 - 0.002380861877349834*x3_1 - 0.00032126052131887958*x4_1 - 1.421502914054107e-5*x5_1 + 4.9891089202995129e-6*x6_1 + 5.7556141375757583e-7*x7_1 - 1.8567670397752609e-8*x8_1)
    var x15_1 = ((((
       1.4981088130721367e-10*exp((-0.63529999999999998)*log(abs(T)))
    )) if truthy((x12_1)) else ((
       3.2867337024382687e-10*x13_1*x14_1
    ))))
    var x16_1 = 7.9674337148168363e-7*exp((-0.5)*log(abs(T)))
    var x17_1 = 1.0e-8*exp((-0.40000000000000002)*log(abs(T)))
    var x18_1 = 1.0/T
    var x19_1 = exp(-457.0*x18_1)
    var x20_1 = 1.0000000000000001e-9*x19_1
    var x21_1 = exp((-0.75)*log(abs(T)))
    var x22_1 = exp(-127500.0*x18_1)
    var x23_1 = T <= 10000.0
    var x24_1 = ((((
       1.26e-9*x21_1*x22_1
    )) if truthy((x23_1)) else ((
       4.0000000000000003e-37*exp((4.7400000000000002)*log(abs(T)))
    ))))
    var x25_1 = exp(-37.100000000000001*x18_1)
    var x26_1 = T >= 50.0
    var x27_1 = ((((
       2.0000000000000001e-10*exp((0.40200000000000002)*log(abs(T)))*x25_1 - 3.3099999999999998e-17*exp((1.48)*log(abs(T)))
    )) if truthy((x26_1)) else ((
       0
    ))))
    var x28_1 = sqrt(T)
    var x29_1 = 1.0/x28_1
    var x30_1 = T >= 10.0  and  T <= 100000.0
    var x31_1 = ((((
       -7.7700000000000002e-13*T + 2.5000000000000002e-10*x28_1 + 2.96e-6*x29_1 - 1.73e-9
    )) if truthy((x30_1)) else ((
       0
    ))))
    var x32_1 = log(T)
    var x33_1 = Log10
    var x34_1 = 1.0/x33_1
    var x35_1 = x32_1*x34_1
    var x36_1 = exp((-3.0)*log(abs(x33_1)))
    var x37_1 = exp((-2.0)*log(abs(x33_1)))
    var x38_1 = ((x32_1)*(x32_1))
    var x39_1 = exp((-0.12690000000000001*((x32_1)*(x32_1)*(x32_1))*x36_1 - 1.5229999999999999*x35_1 + 1.1180000000000001*x37_1*x38_1 - 19.379999999999999)*log(abs(10.0)))
    var x40_1 = vget(X, 5)*x39_1
    var x41_1 = T < 30
    var x42_1 = powi_m3(x33_1)
    var x43_1 = ((x32_1)*(x32_1)*(x32_1))
    var x44_1 = exp((-2)*log(abs(x33_1)))
    var x45_1 = ((x32_1)*(x32_1))
    var x46_1 = exp((-3.194*x35_1 - 0.2072*x42_1*x43_1 + 1.786*x44_1*x45_1 - 18.199999999999999)*log(abs(10.0)))
    var x47_1 = ((((
       3.4977396723747635e-20*exp((-0.14999999999999999)*log(abs(T)))
    )) if truthy((x41_1)) else ((
       x46_1
    ))))
    var x48_1 = exp(-21237.150000000001*x18_1)
    var x49_1 = ((((x32_1)*(x32_1)))*(((x32_1)*(x32_1))))
    var x50_1 = ((x32_1)*(x32_1)*(x32_1)*(x32_1)*(x32_1))
    var x51_1 = exp((6)*log(abs(x32_1)))
    var x52_1 = x48_1*(3.5311931999999998e-13*((x32_1)*(x32_1)*(x32_1)*(x32_1)*(x32_1)*(x32_1)*(x32_1)) + 3.3735381999999997e-7*x32_1 + 3.4172804999999998e-8*x43_1 - 1.4491368e-7*x45_1 - 4.7813727999999997e-9*x49_1 + 3.9731542e-10*x50_1 - 1.8171411000000001e-11*x51_1 - 3.3232183000000002e-7)
    var x53_1 = T >= 100.0  and  T <= 30000.0
    var x54_1 = ((((
       x52_1
    )) if truthy((x53_1)) else ((
       0
    ))))
    var x55_1 = 2.8833736969617052e-16*exp((0.25)*log(abs(T)))
    var x56_1 = exp(-33.0*x18_1)
    var x57_1 = ((((
       2.0299999999999998e-9*exp((-0.33200000000000002)*log(abs(T))) + 2.0600000000000001e-10*exp((0.39600000000000002)*log(abs(T)))*x56_1
    )) if truthy((x26_1)) else ((
       0
    ))))
    var x58_1 = 8.4600000000000008e-10*x35_1 - 1.3700000000000002e-10*x44_1*x45_1 + 4.1700000000000001e-10
    var x59_1 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x60_1 = 2.0860422997526066e-16*x59_1
    var x61_1 = 3.4767371836380304e-16*x59_1
    var x62_1 = exp((-2)*log(abs(T)))
    var x63_1 = x18_1*x32_1
    var x64_1 = x18_1*x34_1
    var x65_1 = x18_1*x45_1
    var x66_1 = x1_1*x18_1
    var x67_1 = x18_1*x3_1
    var x68_1 = x18_1*x7_1
    mset(jac, 1, 0, -vget(X, 1)*x15_1 + vget(X, 2)*x11_1*x9_1)
    mset(jac, 1, 1, -vget(X, 0)*x15_1 - vget(X, 10)*x20_1 - vget(X, 13)*x24_1 - vget(X, 2)*x47_1 - vget(X, 3)*x17_1 - vget(X, 3)*x31_1 - vget(X, 5)*x27_1 - vget(X, 7)*x16_1 - vget(X, 8)*x54_1 - x40_1)
    mset(jac, 1, 2, vget(X, 0)*x11_1*x9_1 - vget(X, 1)*x47_1 + vget(X, 12)*x55_1 + vget(X, 4)*x57_1 + 6.0e-10*vget(X, 6) + 6.3999999999999996e-10*vget(X, 9))
    mset(jac, 1, 3, -vget(X, 1)*x17_1 - vget(X, 1)*x31_1)
    mset(jac, 1, 4, vget(X, 2)*x57_1 + vget(X, 8)*x58_1)
    mset(jac, 1, 5, -vget(X, 1)*x27_1 - vget(X, 1)*x39_1)
    mset(jac, 1, 6, 6.0e-10*vget(X, 2))
    mset(jac, 1, 7, -vget(X, 1)*x16_1)
    mset(jac, 1, 8, -vget(X, 1)*x54_1 + vget(X, 4)*x58_1)
    mset(jac, 1, 9, 6.3999999999999996e-10*vget(X, 2))
    mset(jac, 1, 10, -vget(X, 1)*x20_1)
    mset(jac, 1, 11, 0)
    mset(jac, 1, 12, vget(X, 2)*x55_1)
    mset(jac, 1, 13, -vget(X, 1)*x24_1)
    mset(jac, 1, 14, (3.9837168574084181e-7*exp((-1.5)*log(abs(T)))*vget(X, 1)*vget(X, 7) + 4.0000000000000002e-9*exp((-1.3999999999999999)*log(abs(T)))*vget(X, 1)*vget(X, 3) + 7.997727392299023e-69*exp((12.536555999999999)*log(abs(T)))*vget(X, 0)*vget(X, 2)*x9_1 - vget(X, 0)*vget(X, 1)*((((
       -9.5174852894472843e-11*exp((-1.6353)*log(abs(T)))
    )) if truthy((x12_1)) else ((
       -2.3799651743169991e-10*exp((-1.724112565782685)*log(abs(T)))*x14_1 + 3.2867337024382687e-10*x13_1*x14_1*(-0.0071425856320495021*x18_1*x2_1 - 7.1075145702705346e-5*x18_1*x4_1 + 2.9934653521797078e-5*x18_1*x5_1 + 4.0289298963030308e-6*x18_1*x6_1 - 2.7640217188769353e-8*x18_1*x8_1 - 0.04052089463969382*x66_1 - 0.0012850420852755183*x67_1 - 1.4854136318202087e-7*x68_1)
    )))) + 5.9082438637265071e-70*vget(X, 0)*vget(X, 2)*x10_1*x9_1*(4.6894649399999997*x18_1*x2_1 + 0.1741279885*x18_1*x4_1 - 0.015791857020000001*x18_1*x5_1 + 0.00078368076500000001*x18_1*x6_1 - 11.478657500000001*x66_1 - 1.1508224*x67_1 - 1.6313198799999999e-5*x68_1) - 4.5700000000000003e-7*vget(X, 1)*vget(X, 10)*x19_1*x62_1 - vget(X, 1)*vget(X, 13)*((((
       0.00016065*exp((-2.75)*log(abs(T)))*x22_1 - 9.4499999999999994e-10*exp((-1.75)*log(abs(T)))*x22_1
    )) if truthy((x23_1)) else ((
       1.8960000000000001e-36*exp((3.7400000000000002)*log(abs(T)))
    )))) - vget(X, 1)*vget(X, 2)*((((
       -5.2466095085621454e-21*exp((-1.1499999999999999)*log(abs(T)))
    )) if truthy((x41_1)) else ((
       x33_1*x46_1*(3.5720000000000001*x18_1*x32_1*x44_1 - 0.62159999999999993*x42_1*x65_1 - 3.194*x64_1)
    )))) - vget(X, 1)*vget(X, 3)*((((
       1.2500000000000001e-10*x29_1 - 7.7700000000000002e-13 - 1.48e-6/exp((3.0/2.0)*log(abs(T)))
    )) if truthy((x30_1)) else ((
       0
    )))) - vget(X, 1)*vget(X, 5)*((((
       7.4200000000000004e-9*exp((-1.5979999999999999)*log(abs(T)))*x25_1 + 8.0400000000000002e-11*exp((-0.59799999999999998)*log(abs(T)))*x25_1 - 4.8987999999999998e-17*exp((0.47999999999999998)*log(abs(T)))
    )) if truthy((x26_1)) else ((
       0
    )))) - vget(X, 1)*vget(X, 8)*((((
       x48_1*(-1.9125491199999999e-8*x18_1*x43_1 + 1.9865770999999999e-9*x18_1*x49_1 - 1.09028466e-10*x18_1*x50_1 + 2.4718352399999997e-12*x18_1*x51_1 + 3.3735381999999997e-7*x18_1 - 2.8982736e-7*x63_1 + 1.0251841499999999e-7*x65_1) + 21237.150000000001*x52_1*x62_1
    )) if truthy((x53_1)) else ((
       0
    )))) - vget(X, 1)*x40_1*(5.1485802679346868*x18_1*exp((1.0)*log(abs(x32_1)))*x37_1 - 0.87659414490283338*x18_1*x36_1*x38_1 - 3.5068370966299316*x64_1) + 7.2084342424042629e-17*vget(X, 12)*vget(X, 2)*x21_1 + vget(X, 2)*vget(X, 4)*((((
       6.7980000000000007e-9*exp((-1.6040000000000001)*log(abs(T)))*x56_1 - 6.7396000000000002e-10*exp((-1.3320000000000001)*log(abs(T))) + 8.1576000000000009e-11*exp((-0.60399999999999998)*log(abs(T)))*x56_1
    )) if truthy((x26_1)) else ((
       0
    )))) + vget(X, 4)*vget(X, 8)*(8.4600000000000008e-10*x18_1*x34_1 - 2.7400000000000004e-10*x44_1*x63_1))/(vget(X, 0)*x60_1 + vget(X, 1)*x60_1 + vget(X, 10)*x61_1 + vget(X, 11)*x60_1 + vget(X, 12)*x60_1 + vget(X, 13)*x60_1 + vget(X, 2)*x60_1 + vget(X, 3)*x60_1 + vget(X, 4)*x60_1 + vget(X, 5)*x60_1 + vget(X, 6)*x61_1 + vget(X, 7)*x60_1 + vget(X, 8)*x61_1 + vget(X, 9)*x61_1))
    var x0_2 = sqrt(T)
    var x1_2 = 1.0/x0_2
    var x2_2 = 7.1999999999999996e-8*x1_2
    var x3_2 = exp(-6.1728395061728397e-5*T)
    var x4_2 = vget(X, 2)*x3_2
    var x5_2 = exp((0.92800000000000005)*log(abs(T)))
    var x6_2 = 1.4000000000000001e-18*x5_2
    var x7_2 = 1.0/T
    var x8_2 = exp(-46707.0*x7_2)
    var x9_2 = vget(X, 8)*x8_2
    var x10_2 = 35.5*exp((-2.2799999999999998)*log(abs(T)))
    var x11_2 = exp(-102000.0*x7_2)
    var x12_2 = vget(X, 8)*x11_2
    var x13_2 = 8.7599999999999997e-10*exp((0.34999999999999998)*log(abs(T)))
    var x14_2 = ((T)*(T))
    var x15_2 = ((T)*(T)*(T))
    var x16_2 = ((((T)*(T)))*(((T)*(T))))
    var x17_2 = T <= 10000.0
    var x18_2 = 2*((((
       -5.5279999999999998e-28*((T)*(T)*(T)*(T)*(T)) - 2.3088e-11*T + 7.3427999999999993e-15*x14_2 - 7.5474000000000004e-19*x15_2 + 3.3467999999999999e-23*x16_2 + 4.2277999999999996e-8
    )) if truthy((x17_2)) else ((
       0
    ))))
    var x19_2 = 8.6173430000000006e-5*T
    var x20_2 = log(x19_2)
    var x21_2 = ((x20_2)*(x20_2))
    var x22_2 = ((x20_2)*(x20_2)*(x20_2))
    var x23_2 = ((((x20_2)*(x20_2)))*(((x20_2)*(x20_2))))
    var x24_2 = ((x20_2)*(x20_2)*(x20_2)*(x20_2)*(x20_2))
    var x25_2 = exp((6)*log(abs(x20_2)))
    var x26_2 = ((x20_2)*(x20_2)*(x20_2)*(x20_2)*(x20_2)*(x20_2)*(x20_2))
    var x27_2 = exp((8)*log(abs(x20_2)))
    var x28_2 = exp(-0.28274430617039997*x21_2 + 0.01623316639567*x22_2 - 0.033650120313629989*x23_2 + 0.01178329782711*x24_2 - 0.001656194699504*x25_2 + 0.0001068275202678*x26_2 - 2.6312858092069998e-6*x27_2)
    var x29_2 = vget(X, 3)*x28_2
    var x30_2 = 3.7903999274394518e-18*exp((2.360852208681)*log(abs(T)))
    var x31_2 = x29_2*x30_2
    var x32_2 = exp(-5.7393287500000003*x21_2 + 1.56315498*x22_2 - 0.28770560000000001*x23_2 + 0.034825597700000002*x24_2 - 0.00263197617*x25_2 + 0.000111954395*x26_2 - 2.0391498499999999e-6*x27_2)
    var x33_2 = vget(X, 2)*x32_2
    var x34_2 = 5.9082438637265071e-70*exp((13.536555999999999)*log(abs(T)))
    var x35_2 = x33_2*x34_2
    var x36_2 = x19_2 <= 5500.0
    var x37_2 = exp((-0.72411256578268512)*log(abs(T)))
    var x38_2 = ((x20_2)*(x20_2)*(x20_2)*(x20_2)*(x20_2)*(x20_2)*(x20_2)*(x20_2)*(x20_2))
    var x39_2 = exp(-0.02026044731984691*x21_2 - 0.002380861877349834*x22_2 - 0.00032126052131887958*x23_2 - 1.421502914054107e-5*x24_2 + 4.9891089202995129e-6*x25_2 + 5.7556141375757583e-7*x26_2 - 1.8567670397752609e-8*x27_2 - 3.0711352431965949e-9*x38_2)
    var x40_2 = ((((
       1.4981088130721367e-10*exp((-0.63529999999999998)*log(abs(T)))
    )) if truthy((x36_2)) else ((
       3.2867337024382687e-10*x37_2*x39_2
    ))))
    var x41_2 = exp((-0.75)*log(abs(T)))
    var x42_2 = exp(-127500.0*x7_2)
    var x43_2 = ((((
       1.26e-9*x41_2*x42_2
    )) if truthy((x17_2)) else ((
       4.0000000000000003e-37*exp((4.7400000000000002)*log(abs(T)))
    ))))
    var x44_2 = exp(-37.100000000000001*x7_2)
    var x45_2 = T >= 50.0
    var x46_2 = ((((
       2.0000000000000001e-10*exp((0.40200000000000002)*log(abs(T)))*x44_2 - 3.3099999999999998e-17*exp((1.48)*log(abs(T)))
    )) if truthy((x45_2)) else ((
       0
    ))))
    var x47_2 = exp(-21237.150000000001*x7_2)
    var x48_2 = log(T)
    var x49_2 = ((x48_2)*(x48_2))
    var x50_2 = ((x48_2)*(x48_2)*(x48_2))
    var x51_2 = ((((x48_2)*(x48_2)))*(((x48_2)*(x48_2))))
    var x52_2 = ((x48_2)*(x48_2)*(x48_2)*(x48_2)*(x48_2))
    var x53_2 = exp((6)*log(abs(x48_2)))
    var x54_2 = x47_2*(3.5311931999999998e-13*((x48_2)*(x48_2)*(x48_2)*(x48_2)*(x48_2)*(x48_2)*(x48_2)) + 3.3735381999999997e-7*x48_2 - 1.4491368e-7*x49_2 + 3.4172804999999998e-8*x50_2 - 4.7813727999999997e-9*x51_2 + 3.9731542e-10*x52_2 - 1.8171411000000001e-11*x53_2 - 3.3232183000000002e-7)
    var x55_2 = T >= 100.0  and  T <= 30000.0
    var x56_2 = ((((
       x54_2
    )) if truthy((x55_2)) else ((
       0
    ))))
    var x57_2 = T < 30
    var x58_2 = Log10
    var x59_2 = 1.0/x58_2
    var x60_2 = x48_2*x59_2
    var x61_2 = powi_m3(x58_2)
    var x62_2 = x50_2*x61_2
    var x63_2 = exp((-2)*log(abs(x58_2)))
    var x64_2 = exp((1.786*x49_2*x63_2 - 3.194*x60_2 - 0.2072*x62_2 - 18.199999999999999)*log(abs(10.0)))
    var x65_2 = ((((
       3.4977396723747635e-20*exp((-0.14999999999999999)*log(abs(T)))
    )) if truthy((x57_2)) else ((
       x64_2
    ))))
    var x66_2 = T >= 10.0  and  T <= 100000.0
    var x67_2 = 2*((((
       -7.7700000000000002e-13*T + 2.5000000000000002e-10*x0_2 + 2.96e-6*x1_2 - 1.73e-9
    )) if truthy((x66_2)) else ((
       0
    ))))
    var x68_2 = sqrt(T)
    var x69_1 = 1.0/x68_2
    var x70_1 = vget(X, 1) + vget(X, 10) + vget(X, 2) + vget(X, 3) + 2.0*vget(X, 6) + 2.0*vget(X, 8) + vget(X, 9)
    var x71_1 = x49_2*x63_2
    var x72_1 = -4.8909149999999997*x48_2*x59_2 - 133.82830000000001*x7_2 + 0.47490300000000002*x71_1
    var x73_1 = exp((x72_1 + 14.82123)*log(abs(10.0)))
    var x74_1 = 1.0/x73_1
    var x75_1 = x70_1*x74_1
    var x76_1 = exp(-0.0022727272727272726*T)
    var x77_1 = exp(-0.00054054054054054055*T)
    var x78_1 = -2.0563129999999998*x76_1 + 0.58640729999999996*x77_1 + 0.82274429999999998
    var x79_1 = exp((x78_1)*log(abs(x75_1)))
    var x80_1 = x79_1 + 1.0
    var x81_1 = 1.0/x80_1
    var x82_1 = 16780.950000000001*x7_2 + 1.0
    var x83_1 = 40870.379999999997*x7_2 + 1.0
    var x84_1 = -69.700860000000006*x59_2*log(x83_1) + 4.6331670000000003*x62_2
    var x85_1 = 37.886913*x49_2*x63_2 + 19.734269999999999*x59_2*log(x82_1) - 14.509090000000008*x60_2 - x84_1 - 307.31920000000002
    var x86_1 = exp((x72_1 + 13.656822)*log(abs(10.0)))
    var x87_1 = 1.0/x86_1
    var x88_1 = x70_1*x87_1
    var x89_0 = exp((x78_1)*log(abs(x88_1)))
    var x90_0 = x89_0 + 1.0
    var x91_0 = 1.0/x90_0
    var x92_0 = exp((43.20243*x49_2*x63_2 - 68.422430000000006*x60_2 - 2080.4099999999999*x7_2*x91_0 - 23705.700000000001*x7_2 - x81_1*x85_1 - x84_1 - 178.4239)*log(abs(10.0)))
    var x93_0 = x7_2*x89_0/((x90_0)*(x90_0))
    var x94_0 = 4790.3210533157426*x93_0
    var x95_0 = x86_1*x87_1
    var x96_0 = 1.0/x70_1
    var x97_0 = x78_1*x96_0
    var x98_0 = x95_0*x97_0
    var x99_0 = x79_1*x85_1/((x80_1)*(x80_1))
    var x100_0 = 2.3025850929940459*x99_0
    var x101_0 = x73_1*x74_1
    var x102_0 = x101_0*x97_0
    var x103_0 = x92_0*(x100_0*x102_0 + x94_0*x98_0)
    var x104_0 = -2.4640089999999999*x48_2*x59_2 + 743.05999999999995*x7_2 + 0.19859550000000001*x71_1
    var x105_0 = exp((x104_0 + 9.3055640000000004)*log(abs(10.0)))
    var x106_0 = 1.0/x105_0
    var x107_0 = x106_0*x70_1
    var x108_0 = 2.9375070000000001*x76_1 + 0.23588480000000001*x77_1 + 0.75022860000000002
    var x109_0 = exp((x108_0)*log(abs(x107_0)))
    var x110_0 = x109_0 + 1.0
    var x111_0 = 1.0/x110_0
    var x112_0 = 14254.549999999999*x7_2 + 1.0
    var x113_0 = 27535.310000000001*x7_2 + 1.0
    var x114_0 = -21.360939999999999*x59_2*log(x113_0) + 0.25820969999999999*x62_2
    var x115_0 = -x114_0 + 70.138370000000009*x48_2*x59_2 + 11.28215*x59_2*log(x112_0) - 4.7035149999999994*x71_1 - 203.11568
    var x116_0 = exp((x104_0 + 8.1313220000000008)*log(abs(10.0)))
    var x117_0 = 1.0/x116_0
    var x118_0 = x117_0*x70_1
    var x119_0 = exp((x108_0)*log(abs(x118_0)))
    var x120_0 = x119_0 + 1.0
    var x121_0 = 1.0/x120_0
    var x122_0 = exp((-x111_0*x115_0 - x114_0 - 1657.4099999999999*x121_0*x7_2 + 42.707410000000003*x48_2*x59_2 - 21467.790000000001*x7_2 - 2.0273650000000001*x71_1 - 142.7664)*log(abs(10.0)))
    var x123_0 = x119_0*x7_2/((x120_0)*(x120_0))
    var x124_0 = 3816.3275589792611*x123_0
    var x125_0 = x116_0*x117_0
    var x126_0 = x108_0*x96_0
    var x127_0 = x125_0*x126_0
    var x128_0 = x109_0*x115_0/((x110_0)*(x110_0))
    var x129_0 = 2.3025850929940459*x128_0
    var x130_0 = x105_0*x106_0
    var x131_0 = x126_0*x130_0
    var x132_0 = x122_0*(x124_0*x127_0 + x129_0*x131_0)
    var x133_0 = -x103_0 - x132_0
    var x134_0 = vget(X, 2)*vget(X, 8)
    var x135_0 = 3*x103_0 + 3*x132_0
    var x136_0 = log(0.0001*T)
    var x137_0 = exp((-1.6200000000000001*((x136_0)*(x136_0))*x63_2 + 1.3*x136_0*x59_2 - 4.8449999999999998)*log(abs(10.0)))
    var x138_0 = x137_0*x70_1
    var x139_0 = x138_0 + 1.0
    var x140_0 = exp((-2)*log(abs(x139_0)))
    var x141_0 = 1.0 - exp(-6000.0*x7_2)
    var x142_0 = 52000.0*x7_2
    var x143_0 = exp(-x142_0)
    var x144_0 = x141_0*x143_0
    var x145_0 = 8.1250000000000003e-8*x144_0*x69_1
    var x146_0 = log(x145_0)
    var x147_0 = x140_0*x146_0
    var x148_0 = ((vget(X, 8))*(vget(X, 8)))
    var x149_0 = 1.1800000000000001e-10*exp(-69500.0*x7_2)
    var x150_0 = 1.0/x139_0
    var x151_0 = 1.0*x150_0
    var x152_0 = exp((x151_0)*log(abs(x149_0)))
    var x153_0 = 1.0 - x151_0
    var x154_0 = exp((x153_0)*log(abs(x145_0)))
    var x155_0 = x152_0*x154_0
    var x156_0 = x148_0*x155_0
    var x157_0 = x137_0*x156_0
    var x158_0 = 2.0*x157_0
    var x159_0 = x140_0*log(x149_0)
    var x160_0 = x158_0*x159_0
    var x161_0 = x133_0*x134_0 + x134_0*x135_0 + x147_0*x158_0 - x160_0
    var x162_0 = 2.6534040307116387e-9*exp((-0.10000000000000001)*log(abs(T)))
    var x163_0 = exp((0.25)*log(abs(T)))
    var x164_0 = 2.8833736969617052e-16*x163_0
    var x165_0 = 6.1739095063118665e-10*exp((0.40999999999999998)*log(abs(T)))
    var x166_0 = 1.0/x163_0
    var x167_0 = -1.5e-32*x166_0 - 5.0000000000000004e-32*x69_1
    var x168_0 = ((vget(X, 2))*(vget(X, 2)))
    var x169_0 = 1.0/x14_2
    var x170_0 = 5.25e-11*exp(173900.0*x169_0 - 4430.0*x7_2)
    var x171_0 = T > 200.0
    var x172_0 = ((((
       x170_0
    )) if truthy((x171_0)) else ((
       0
    ))))
    var x173_0 = exp(-33.0*x7_2)
    var x174_0 = ((((
       2.0299999999999998e-9*exp((-0.33200000000000002)*log(abs(T))) + 2.0600000000000001e-10*exp((0.39600000000000002)*log(abs(T)))*x173_0
    )) if truthy((x45_2)) else ((
       0
    ))))
    var x175_0 = exp((-3.0)*log(abs(x58_2)))
    var x176_0 = exp((-2.0)*log(abs(x58_2)))
    var x177_0 = ((x48_2)*(x48_2))
    var x178_0 = exp((-0.12690000000000001*x175_0*((x48_2)*(x48_2)*(x48_2)) + 1.1180000000000001*x176_0*x177_0 - 1.5229999999999999*x60_2 - 19.379999999999999)*log(abs(10.0)))
    var x179_0 = vget(X, 4)*x178_0
    var x180_0 = 0.0061910000000000003*exp((1.0461)*log(abs(T))) + 8.9711999999999997e-11*exp((3.0424000000000002)*log(abs(T))) + 3.2575999999999999e-14*exp((3.7740999999999998)*log(abs(T))) + 1.0
    var x181_0 = 1.0/x180_0
    var x182_0 = 1.3500000000000001e-9*exp((0.098492999999999997)*log(abs(T))) + 4.4350199999999998e-10*exp((0.55610000000000004)*log(abs(T))) + 3.7408500000000004e-16*exp((2.1825999999999999)*log(abs(T)))
    var x183_0 = x181_0*x182_0
    var x184_0 = T <= 1160.0
    var x185_0 = exp(-0.14210135215541481*x21_2 + 0.0084644553866299998*x22_2 - 0.0014327641212992001*x23_2 + 0.00020122502847909999*x24_2 + 8.6639632430900003e-5*x25_2 - 2.5850096802639999e-5*x26_2 + 2.4555011970391999e-6*x27_2 - 8.0683824611800006e-8*x38_2)
    var x186_0 = 3.3178155742407614e-14*exp((1.1394493358416311)*log(abs(T)))*x185_0
    var x187_0 = ((((
       1.4643482606109061e-16*exp((1.78186)*log(abs(T)))
    )) if truthy((x184_0)) else ((
       x186_0
    ))))
    var x188_0 = -x122_0 - x92_0
    var x189_0 = 3*x92_0
    var x190_0 = 3*x122_0
    var x191_0 = x189_0 + x190_0
    var x192_0 = 4.9999999999999996e-6*x1_2
    var x193_0 = powi_m5(x58_2)
    var x194_0 = exp((-4)*log(abs(x58_2)))
    var x195_0 = exp((0.31788699999999998*x193_0*x52_2 - 2.1690299999999998*x194_0*x51_2 + 5.8888600000000002*x60_2 + 2.2506900000000001*x62_2 + 7.1969200000000004*x71_1 - 56.473700000000001)*log(abs(10.0)))
    var x196_0 = T <= 1167.4796423742259
    var x197_0 = exp(-5207.0*x7_2)
    var x198_0 = ((((
       x195_0
    )) if truthy((x196_0)) else ((
       3.1699999999999999e-10*x197_0
    ))))
    var x199_0 = 4.6051701859880918*x102_0*x99_0 + 9580.6421066314851*x93_0*x98_0
    var x200_0 = 7632.6551179585222*x123_0*x127_0 + 4.6051701859880918*x128_0*x131_0
    var x201_0 = 4.0*x157_0
    var x202_0 = x134_0*(-x122_0*x200_0 - x199_0*x92_0) + x134_0*(x189_0*x199_0 + x190_0*x200_0) + x147_0*x201_0 - x159_0*x201_0
    var x203_0 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x204_0 = 2.0860422997526066e-16*x203_0
    var x205_0 = 3.4767371836380304e-16*x203_0
    var x206_0 = exp((-1.5)*log(abs(T)))
    var x207_0 = vget(X, 2)*vget(X, 7)
    var x208_0 = 2.5313028975878652e-10*exp((-0.59000000000000008)*log(abs(T)))
    var x209_0 = exp((-3.0/2.0)*log(abs(T)))
    var x210_0 = vget(X, 0)*x4_2
    var x211_0 = vget(X, 0)*x9_2
    var x212_0 = vget(X, 0)*x12_2
    var x213_0 = ((vget(X, 2))*(vget(X, 2))*(vget(X, 2)))
    var x214_0 = exp((-1.25)*log(abs(T)))
    var x215_0 = vget(X, 2)*vget(X, 3)
    var x216_0 = x59_2*x7_2
    var x217_0 = x49_2*x7_2
    var x218_0 = x217_0*x61_2
    var x219_0 = x48_2*x7_2
    var x220_0 = x50_2*x7_2
    var x221_0 = x51_2*x7_2
    var x222_0 = x20_2*x7_2
    var x223_0 = x22_2*x7_2
    var x224_0 = x24_2*x7_2
    var x225_0 = x26_2*x7_2
    var x226_0 = x219_0*x63_2
    var x227_0 = x27_2*x7_2
    var x228_0 = 1.0*x138_0*(-7.460375701300709*x136_0*x63_2*x7_2 + 2.9933606208922598*x59_2*x7_2)
    var x229_0 = 2*x156_0
    var x230_0 = exp((-2.5)*log(abs(T)))
    var x231_0 = x169_0*x59_2
    var x232_0 = x231_0/x83_1
    var x233_0 = 0.0046734386363636356*x76_1 - 0.00031697691891891889*x77_1
    var x234_0 = x78_1*(-308.15104860073512*x169_0 - 2.1870091368363029*x226_0 + 11.261747970100974*x59_2*x7_2)
    var x235_0 = x100_0*(x101_0*x234_0 + x233_0*log(x75_1)) + 4790.3210533157426*x169_0*x91_0 + 54584.391438988954*x169_0 - 157.54846734442862*x216_0 - 32.004783802655837*x218_0 + 198.95454259823751*x226_0 - 6559375.6154640894*x232_0 - 2.3025850929940459*x81_1*(-14.509090000000008*x216_0 - 13.899501000000001*x218_0 - 331159.79815649998*x231_0/x82_1 - 2848700.6345267999*x232_0 + 75.773826*x48_2*x63_2*x7_2) + x94_0*(x233_0*log(x88_1) + x234_0*x95_0)
    var x236_0 = x231_0/x113_0
    var x237_0 = -0.0066761522727272725*x76_1 - 0.0001275052972972973*x77_1
    var x238_0 = x108_0*(1710.9588792001557*x169_0 + 5.6735903924031659*x216_0 - 0.91456607567139814*x226_0)
    var x239_0 = -2.3025850929940459*x111_0*(-0.77462909999999996*x218_0 - 9.4070299999999989*x226_0 - 588180.10479140002*x236_0 + 70.138370000000009*x59_2*x7_2 - 160821.97128249999*x231_0/x112_0) + 3816.3275589792611*x121_0*x169_0 + x124_0*(x125_0*x238_0 + x237_0*log(x118_0)) + x129_0*(x130_0*x238_0 + x237_0*log(x107_0)) + 49431.413233526648*x169_0 + 98.337445626384849*x216_0 - 1.783649418259394*x218_0 - 9.3363608541157479*x226_0 - 1354334.7412883535*x236_0
    mset(jac, 2, 0, vget(X, 1)*x40_2 + vget(X, 6)*x18_2 + vget(X, 9)*x2_2 + x10_2*x9_2 + x12_2*x13_2 + x31_2 - x35_2 - x4_2*x6_2)
    mset(jac, 2, 1, vget(X, 0)*x40_2 + vget(X, 13)*x43_2 - vget(X, 2)*x65_2 + vget(X, 3)*x67_2 + vget(X, 5)*x46_2 + 7.9674337148168363e-7*vget(X, 7)*x69_1 + vget(X, 8)*x56_2 + x161_0)
    mset(jac, 2, 2, -vget(X, 0)*x3_2*x6_2 - vget(X, 0)*x32_2*x34_2 - vget(X, 1)*x65_2 - vget(X, 10)*x172_0 - vget(X, 12)*x164_0 + vget(X, 2)*vget(X, 8)*x133_0 + vget(X, 2)*vget(X, 8)*x135_0 + 2*vget(X, 2)*vget(X, 8)*x167_0 - vget(X, 3)*x183_0 + vget(X, 3)*x187_0 - vget(X, 4)*x174_0 - 1.0e-25*vget(X, 5) - 6.0e-10*vget(X, 6) - vget(X, 7)*x162_0 - vget(X, 7)*x165_0 + vget(X, 8)*x188_0 + vget(X, 8)*x191_0 - 6.3999999999999996e-10*vget(X, 9) + 2.0*x137_0*x140_0*x146_0*x148_0*x152_0*x154_0 - x160_0 + 3*x168_0*(-1.8e-31*x166_0 - 6.0000000000000005e-31*x69_1) + 3*x168_0*(6.0000000000000001e-32*x166_0 + 2.0000000000000002e-31*x69_1) - x179_0)
    mset(jac, 2, 3, vget(X, 0)*x28_2*x30_2 + vget(X, 1)*x67_2 - vget(X, 2)*x183_0 + vget(X, 2)*x187_0 + vget(X, 5)*x165_0 + vget(X, 6)*x192_0 + x161_0)
    mset(jac, 2, 4, -vget(X, 2)*x174_0 - vget(X, 2)*x178_0)
    mset(jac, 2, 5, vget(X, 1)*x46_2 - 1.0e-25*vget(X, 2) + vget(X, 3)*x165_0 + vget(X, 8)*x198_0)
    mset(jac, 2, 6, vget(X, 0)*x18_2 - 6.0e-10*vget(X, 2) + vget(X, 3)*x192_0 + x202_0)
    mset(jac, 2, 7, 7.9674337148168363e-7*vget(X, 1)*x69_1 - vget(X, 2)*x162_0 - vget(X, 2)*x165_0)
    mset(jac, 2, 8, vget(X, 0)*x10_2*x8_2 + vget(X, 0)*x11_2*x13_2 + vget(X, 1)*x56_2 + vget(X, 2)*x188_0 + vget(X, 2)*x191_0 + vget(X, 5)*x198_0 + 4*vget(X, 8)*x155_0 + x167_0*x168_0 + x202_0)
    mset(jac, 2, 9, vget(X, 0)*x2_2 - 6.3999999999999996e-10*vget(X, 2) + x161_0)
    mset(jac, 2, 10, -vget(X, 2)*x172_0 + x161_0)
    mset(jac, 2, 11, 0)
    mset(jac, 2, 12, -vget(X, 2)*x164_0)
    mset(jac, 2, 13, vget(X, 1)*x43_2)
    mset(jac, 2, 14, (1658098.5*exp((-4.2799999999999994)*log(abs(T)))*x211_0 - 80.939999999999998*exp((-3.2799999999999998)*log(abs(T)))*x211_0 + 8.9351999999999994e-5*exp((-1.6499999999999999)*log(abs(T)))*x212_0 + 2.6534040307116389e-10*exp((-1.1000000000000001)*log(abs(T)))*x207_0 + 3.0659999999999995e-10*exp((-0.65000000000000002)*log(abs(T)))*x212_0 - 1.2992000000000002e-18*exp((-0.071999999999999953)*log(abs(T)))*x210_0 + 8.9485740404797324e-18*exp((1.360852208681)*log(abs(T)))*vget(X, 0)*x29_2 - 7.997727392299023e-69*exp((12.536555999999999)*log(abs(T)))*vget(X, 0)*x33_2 + vget(X, 0)*vget(X, 1)*((((
       -9.5174852894472843e-11*exp((-1.6353)*log(abs(T)))
    )) if truthy((x36_2)) else ((
       -2.3799651743169991e-10*exp((-1.724112565782685)*log(abs(T)))*x39_2 + 3.2867337024382687e-10*x37_2*x39_2*(-0.0071425856320495021*x21_2*x7_2 - 0.04052089463969382*x222_0 - 0.0012850420852755183*x223_0 - 1.4854136318202087e-7*x225_0 - 2.7640217188769353e-8*x227_0 - 7.1075145702705346e-5*x23_2*x7_2 + 2.9934653521797078e-5*x24_2*x7_2 + 4.0289298963030308e-6*x25_2*x7_2)
    )))) + 2*vget(X, 0)*vget(X, 6)*((((
       1.4685599999999999e-14*T - 2.2642200000000001e-18*x14_2 + 1.3387199999999999e-22*x15_2 - 2.7639999999999999e-27*x16_2 - 2.3088e-11
    )) if truthy((x17_2)) else ((
       0
    )))) - 3.5999999999999998e-8*vget(X, 0)*vget(X, 9)*x209_0 + vget(X, 0)*x31_2*(0.048699499187009998*x21_2*x7_2 - 0.56548861234079995*x222_0 - 0.13460048125451995*x223_0 - 0.009937168197024001*x224_0 - 2.1050286473655998e-5*x225_0 + 0.058916489135550004*x23_2*x7_2 + 0.00074779264187460007*x25_2*x7_2) - vget(X, 0)*x35_2*(4.6894649399999997*x21_2*x7_2 - 11.478657500000001*x222_0 - 1.1508224*x223_0 - 0.015791857020000001*x224_0 - 1.6313198799999999e-5*x225_0 + 0.1741279885*x23_2*x7_2 + 0.00078368076500000001*x25_2*x7_2) + vget(X, 1)*vget(X, 13)*((((
       0.00016065*exp((-2.75)*log(abs(T)))*x42_2 - 9.4499999999999994e-10*exp((-1.75)*log(abs(T)))*x42_2
    )) if truthy((x17_2)) else ((
       1.8960000000000001e-36*exp((3.7400000000000002)*log(abs(T)))
    )))) - vget(X, 1)*vget(X, 2)*((((
       -5.2466095085621454e-21*exp((-1.1499999999999999)*log(abs(T)))
    )) if truthy((x57_2)) else ((
       x58_2*x64_2*(-3.194*x216_0 - 0.62159999999999993*x218_0 + 3.5720000000000001*x48_2*x63_2*x7_2)
    )))) + 2*vget(X, 1)*vget(X, 3)*((((
       1.2500000000000001e-10*x1_2 - 1.48e-6*x209_0 - 7.7700000000000002e-13
    )) if truthy((x66_2)) else ((
       0
    )))) + vget(X, 1)*vget(X, 5)*((((
       7.4200000000000004e-9*exp((-1.5979999999999999)*log(abs(T)))*x44_2 + 8.0400000000000002e-11*exp((-0.59799999999999998)*log(abs(T)))*x44_2 - 4.8987999999999998e-17*exp((0.47999999999999998)*log(abs(T)))
    )) if truthy((x45_2)) else ((
       0
    )))) - 3.9837168574084181e-7*vget(X, 1)*vget(X, 7)*x206_0 + vget(X, 1)*vget(X, 8)*((((
       21237.150000000001*x169_0*x54_2 + x47_2*(1.0251841499999999e-7*x217_0 - 2.8982736e-7*x219_0 - 1.9125491199999999e-8*x220_0 + 1.9865770999999999e-9*x221_0 - 1.09028466e-10*x52_2*x7_2 + 2.4718352399999997e-12*x53_2*x7_2 + 3.3735381999999997e-7*x7_2)
    )) if truthy((x55_2)) else ((
       0
    )))) - vget(X, 10)*vget(X, 2)*((((
       x170_0*(4430.0*x169_0 - 347800.0/x15_2)
    )) if truthy((x171_0)) else ((
       0
    )))) - 7.2084342424042629e-17*vget(X, 12)*vget(X, 2)*x41_2 - vget(X, 2)*vget(X, 4)*((((
       6.7980000000000007e-9*exp((-1.6040000000000001)*log(abs(T)))*x173_0 - 6.7396000000000002e-10*exp((-1.3320000000000001)*log(abs(T))) + 8.1576000000000009e-11*exp((-0.60399999999999998)*log(abs(T)))*x173_0
    )) if truthy((x45_2)) else ((
       0
    )))) - vget(X, 2)*x179_0*(-0.87659414490283338*x175_0*x177_0*x7_2 + 5.1485802679346868*x176_0*exp((1.0)*log(abs(x48_2)))*x7_2 - 3.5068370966299316*x216_0) + vget(X, 3)*vget(X, 5)*x208_0 - 2.4999999999999998e-6*vget(X, 3)*vget(X, 6)*x209_0 + vget(X, 5)*vget(X, 8)*((((
       x195_0*x58_2*(1.5894349999999999*x193_0*x221_0 - 8.6761199999999992*x194_0*x220_0 + 5.8888600000000002*x216_0 + 6.7520699999999998*x218_0 + 14.393840000000001*x226_0)
    )) if truthy((x196_0)) else ((
       1.650619e-6*x169_0*x197_0
    )))) + vget(X, 8)*x168_0*(2.5000000000000002e-32*x206_0 + 3.75e-33*x214_0) + x134_0*(-x122_0*x239_0 - x235_0*x92_0) + x134_0*(x189_0*x235_0 + x190_0*x239_0) - x181_0*x215_0*(1.3296555000000001e-10*exp((-0.90150700000000006)*log(abs(T))) + 2.466314622e-10*exp((-0.44389999999999996)*log(abs(T))) + 8.1647792100000001e-16*exp((1.1825999999999999)*log(abs(T)))) - x207_0*x208_0 + 8.6419753086419757e-23*x210_0*x5_2 + x213_0*(-1.0000000000000001e-31*x206_0 - 1.5e-32*x214_0) + x213_0*(3.0000000000000003e-31*x206_0 + 4.5e-32*x214_0) + x215_0*((((
       2.6092635916521491e-16*exp((0.78186)*log(abs(T)))
    )) if truthy((x184_0)) else ((
       3.7804827525136553e-14*exp((0.13944933584163111)*log(abs(T)))*x185_0 + x186_0*(0.025393366159889998*x21_2*x7_2 - 0.28420270431082961*x222_0 - 0.0057310564851968003*x223_0 - 7.2615442150620009e-7*x227_0 + 0.0010061251423955*x23_2*x7_2 + 0.00051983779458540007*x24_2*x7_2 - 0.00018095067761848*x25_2*x7_2 + 1.9644009576313599e-5*x26_2*x7_2)
    )))) + x229_0*(x147_0*x228_0 + 12307692.307692308*x153_0*x68_2*(0.0042250000000000005*x141_0*x143_0*x230_0 - 4.0625000000000001e-8*x144_0*x206_0 - 0.00048750000000000003*x230_0*exp(-58000.0*x7_2))*exp(x142_0)/x141_0) + x229_0*(69500.0*x150_0*x169_0 - x159_0*x228_0) - x182_0*x215_0*(-0.0064764051000000007*exp((0.04610000000000003)*log(abs(T))) - 2.7293978880000002e-10*exp((2.0424000000000002)*log(abs(T))) - 1.229450816e-13*exp((2.7740999999999998)*log(abs(T))))/((x180_0)*(x180_0)))/(vget(X, 0)*x204_0 + vget(X, 1)*x204_0 + vget(X, 10)*x205_0 + vget(X, 11)*x204_0 + vget(X, 12)*x204_0 + vget(X, 13)*x204_0 + vget(X, 2)*x204_0 + vget(X, 3)*x204_0 + vget(X, 4)*x204_0 + vget(X, 5)*x204_0 + vget(X, 6)*x205_0 + vget(X, 7)*x204_0 + vget(X, 8)*x205_0 + vget(X, 9)*x205_0))
    var x0_3 = exp(-6.1728395061728397e-5*T)
    var x1_3 = vget(X, 2)*x0_3
    var x2_3 = exp((0.92800000000000005)*log(abs(T)))
    var x3_3 = 1.4000000000000001e-18*x2_3
    var x4_3 = 1.0/T
    var x5_3 = exp(-46707.0*x4_3)
    var x6_3 = vget(X, 8)*x5_3
    var x7_3 = 35.5*exp((-2.2799999999999998)*log(abs(T)))
    var x8_3 = log(8.6173430000000006e-5*T)
    var x9_3 = ((x8_3)*(x8_3))
    var x10_3 = ((x8_3)*(x8_3)*(x8_3))
    var x11_3 = ((((x8_3)*(x8_3)))*(((x8_3)*(x8_3))))
    var x12_3 = ((x8_3)*(x8_3)*(x8_3)*(x8_3)*(x8_3))
    var x13_3 = exp((6)*log(abs(x8_3)))
    var x14_3 = ((x8_3)*(x8_3)*(x8_3)*(x8_3)*(x8_3)*(x8_3)*(x8_3))
    var x15_3 = exp((8)*log(abs(x8_3)))
    var x16_3 = exp(0.01623316639567*x10_3 - 0.033650120313629989*x11_3 + 0.01178329782711*x12_3 - 0.001656194699504*x13_3 + 0.0001068275202678*x14_3 - 2.6312858092069998e-6*x15_3 - 0.28274430617039997*x9_3)
    var x17_3 = vget(X, 3)*x16_3
    var x18_3 = 3.7903999274394518e-18*exp((2.360852208681)*log(abs(T)))
    var x19_3 = x17_3*x18_3
    var x20_3 = 1.0e-8*exp((-0.40000000000000002)*log(abs(T)))
    var x21_3 = sqrt(T)
    var x22_3 = 1.0/x21_3
    var x23_3 = T >= 10.0  and  T <= 100000.0
    var x24_3 = ((((
       -7.7700000000000002e-13*T + 2.5000000000000002e-10*x21_3 + 2.96e-6*x22_3 - 1.73e-9
    )) if truthy((x23_3)) else ((
       0
    ))))
    var x25_3 = 6.1739095063118665e-10*exp((0.40999999999999998)*log(abs(T)))
    var x26_3 = 0.0061910000000000003*exp((1.0461)*log(abs(T))) + 8.9711999999999997e-11*exp((3.0424000000000002)*log(abs(T))) + 3.2575999999999999e-14*exp((3.7740999999999998)*log(abs(T))) + 1.0
    var x27_3 = 1.0/x26_3
    var x28_3 = 1.3500000000000001e-9*exp((0.098492999999999997)*log(abs(T))) + 4.4350199999999998e-10*exp((0.55610000000000004)*log(abs(T))) + 3.7408500000000004e-16*exp((2.1825999999999999)*log(abs(T)))
    var x29_3 = x27_3*x28_3
    var x30_3 = T <= 1160.0
    var x31_3 = exp(0.0084644553866299998*x10_3 - 0.0014327641212992001*x11_3 + 0.00020122502847909999*x12_3 + 8.6639632430900003e-5*x13_3 - 2.5850096802639999e-5*x14_3 + 2.4555011970391999e-6*x15_3 - 8.0683824611800006e-8*((x8_3)*(x8_3)*(x8_3)*(x8_3)*(x8_3)*(x8_3)*(x8_3)*(x8_3)*(x8_3)) - 0.14210135215541481*x9_3)
    var x32_3 = 3.3178155742407614e-14*exp((1.1394493358416311)*log(abs(T)))*x31_3
    var x33_3 = ((((
       1.4643482606109061e-16*exp((1.78186)*log(abs(T)))
    )) if truthy((x30_3)) else ((
       x32_3
    ))))
    var x34_3 = 2.6534040307116387e-9*exp((-0.10000000000000001)*log(abs(T)))
    var x35_3 = 4.9999999999999996e-6*x22_3
    var x36_3 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x37_3 = 2.0860422997526066e-16*x36_3
    var x38_3 = 3.4767371836380304e-16*x36_3
    var x39_3 = exp((-0.59000000000000008)*log(abs(T)))
    var x40_3 = exp((-3.0/2.0)*log(abs(T)))
    var x41_3 = vget(X, 2)*vget(X, 3)
    var x42_3 = x4_3*x8_3
    var x43_3 = x10_3*x4_3
    mset(jac, 3, 0, x1_3*x3_3 - x19_3 + x6_3*x7_3)
    mset(jac, 3, 1, -vget(X, 3)*x20_3 - vget(X, 3)*x24_3)
    mset(jac, 3, 2, vget(X, 0)*x0_3*x3_3 - vget(X, 3)*x29_3 - vget(X, 3)*x33_3 + vget(X, 7)*x25_3)
    mset(jac, 3, 3, -vget(X, 0)*x16_3*x18_3 - vget(X, 1)*x20_3 - vget(X, 1)*x24_3 - vget(X, 2)*x29_3 - vget(X, 2)*x33_3 - vget(X, 5)*x25_3 - vget(X, 5)*x34_3 - vget(X, 6)*x35_3)
    mset(jac, 3, 4, 0)
    mset(jac, 3, 5, -vget(X, 3)*x25_3 - vget(X, 3)*x34_3)
    mset(jac, 3, 6, -vget(X, 3)*x35_3)
    mset(jac, 3, 7, vget(X, 2)*x25_3)
    mset(jac, 3, 8, vget(X, 0)*x5_3*x7_3)
    mset(jac, 3, 9, 0)
    mset(jac, 3, 10, 0)
    mset(jac, 3, 11, 0)
    mset(jac, 3, 12, 0)
    mset(jac, 3, 13, 0)
    mset(jac, 3, 14, (1658098.5*exp((-4.2799999999999994)*log(abs(T)))*vget(X, 0)*vget(X, 8)*x5_3 - 80.939999999999998*exp((-3.2799999999999998)*log(abs(T)))*vget(X, 0)*x6_3 + 4.0000000000000002e-9*exp((-1.3999999999999999)*log(abs(T)))*vget(X, 1)*vget(X, 3) + 2.6534040307116389e-10*exp((-1.1000000000000001)*log(abs(T)))*vget(X, 3)*vget(X, 5) + 1.2992000000000002e-18*exp((-0.071999999999999953)*log(abs(T)))*vget(X, 0)*vget(X, 2)*x0_3 - 8.9485740404797324e-18*exp((1.360852208681)*log(abs(T)))*vget(X, 0)*x17_3 - 8.6419753086419757e-23*vget(X, 0)*x1_3*x2_3 - vget(X, 0)*x19_3*(0.058916489135550004*x11_3*x4_3 - 0.009937168197024001*x12_3*x4_3 + 0.00074779264187460007*x13_3*x4_3 - 2.1050286473655998e-5*x14_3*x4_3 + 0.048699499187009998*x4_3*x9_3 - 0.56548861234079995*x42_3 - 0.13460048125451995*x43_3) - vget(X, 1)*vget(X, 3)*((((
       1.2500000000000001e-10*x22_3 - 1.48e-6*x40_3 - 7.7700000000000002e-13
    )) if truthy((x23_3)) else ((
       0
    )))) + 2.5313028975878652e-10*vget(X, 2)*vget(X, 7)*x39_3 - 2.5313028975878652e-10*vget(X, 3)*vget(X, 5)*x39_3 + 2.4999999999999998e-6*vget(X, 3)*vget(X, 6)*x40_3 - x27_3*x41_3*(1.3296555000000001e-10*exp((-0.90150700000000006)*log(abs(T))) + 2.466314622e-10*exp((-0.44389999999999996)*log(abs(T))) + 8.1647792100000001e-16*exp((1.1825999999999999)*log(abs(T)))) - x41_3*((((
       2.6092635916521491e-16*exp((0.78186)*log(abs(T)))
    )) if truthy((x30_3)) else ((
       3.7804827525136553e-14*exp((0.13944933584163111)*log(abs(T)))*x31_3 + x32_3*(0.0010061251423955*x11_3*x4_3 + 0.00051983779458540007*x12_3*x4_3 - 0.00018095067761848*x13_3*x4_3 + 1.9644009576313599e-5*x14_3*x4_3 - 7.2615442150620009e-7*x15_3*x4_3 + 0.025393366159889998*x4_3*x9_3 - 0.28420270431082961*x42_3 - 0.0057310564851968003*x43_3)
    )))) - x28_3*x41_3*(-0.0064764051000000007*exp((0.04610000000000003)*log(abs(T))) - 2.7293978880000002e-10*exp((2.0424000000000002)*log(abs(T))) - 1.229450816e-13*exp((2.7740999999999998)*log(abs(T))))/((x26_3)*(x26_3)))/(vget(X, 0)*x37_3 + vget(X, 1)*x37_3 + vget(X, 10)*x38_3 + vget(X, 11)*x37_3 + vget(X, 12)*x37_3 + vget(X, 13)*x37_3 + vget(X, 2)*x37_3 + vget(X, 3)*x37_3 + vget(X, 4)*x37_3 + vget(X, 5)*x37_3 + vget(X, 6)*x38_3 + vget(X, 7)*x37_3 + vget(X, 8)*x38_3 + vget(X, 9)*x38_3))
    var x0_4 = 2.5950363272655348e-10*exp((-0.75)*log(abs(T)))
    var x1_4 = 1.0/T
    var x2_4 = exp(-457.0*x1_4)
    var x3_4 = 1.0000000000000001e-9*x2_4
    var x4_4 = exp(-37.100000000000001*x1_4)
    var x5_4 = T >= 50.0
    var x6_4 = ((((
       2.0000000000000001e-10*exp((0.40200000000000002)*log(abs(T)))*x4_4 - 3.3099999999999998e-17*exp((1.48)*log(abs(T)))
    )) if truthy((x5_4)) else ((
       0
    ))))
    var x7_4 = exp(-33.0*x1_4)
    var x8_4 = ((((
       2.0299999999999998e-9*exp((-0.33200000000000002)*log(abs(T))) + 2.0600000000000001e-10*exp((0.39600000000000002)*log(abs(T)))*x7_4
    )) if truthy((x5_4)) else ((
       0
    ))))
    var x9_4 = Log10
    var x10_4 = 1.0/x9_4
    var x11_4 = log(T)
    var x12_4 = x10_4*x11_4
    var x13_4 = exp((-3.0)*log(abs(x9_4)))
    var x14_4 = exp((-2.0)*log(abs(x9_4)))
    var x15_4 = ((x11_4)*(x11_4))
    var x16_4 = exp((-0.12690000000000001*((x11_4)*(x11_4)*(x11_4))*x13_4 - 1.5229999999999999*x12_4 + 1.1180000000000001*x14_4*x15_4 - 19.379999999999999)*log(abs(10.0)))
    var x17_4 = vget(X, 4)*x16_4
    var x18_4 = 9.8726896031426014e-7*exp((-0.5)*log(abs(T)))
    var x19_4 = exp((-2)*log(abs(x9_4)))
    var x20_4 = 1.3700000000000002e-10*((x11_4)*(x11_4))*x19_4 - 8.4600000000000008e-10*x12_4 - 4.1700000000000001e-10
    var x21_4 = x1_4*x10_4
    var x22_4 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x23_4 = 2.0860422997526066e-16*x22_4
    var x24_4 = 3.4767371836380304e-16*x22_4
    mset(jac, 4, 0, -vget(X, 4)*x0_4)
    mset(jac, 4, 1, vget(X, 10)*x3_4 + vget(X, 5)*x6_4)
    mset(jac, 4, 2, -vget(X, 4)*x8_4 - x17_4)
    mset(jac, 4, 3, 0)
    mset(jac, 4, 4, -vget(X, 0)*x0_4 - vget(X, 2)*x16_4 - vget(X, 2)*x8_4 - vget(X, 7)*x18_4 + vget(X, 8)*x20_4)
    mset(jac, 4, 5, vget(X, 1)*x6_4)
    mset(jac, 4, 6, 0)
    mset(jac, 4, 7, -vget(X, 4)*x18_4)
    mset(jac, 4, 8, vget(X, 4)*x20_4)
    mset(jac, 4, 9, 0)
    mset(jac, 4, 10, vget(X, 1)*x3_4)
    mset(jac, 4, 11, 0)
    mset(jac, 4, 12, 0)
    mset(jac, 4, 13, 0)
    mset(jac, 4, 14, (1.9462772454491511e-10*exp((-1.75)*log(abs(T)))*vget(X, 0)*vget(X, 4) + 4.9363448015713007e-7*exp((-1.5)*log(abs(T)))*vget(X, 4)*vget(X, 7) + vget(X, 1)*vget(X, 5)*((((
       7.4200000000000004e-9*exp((-1.5979999999999999)*log(abs(T)))*x4_4 + 8.0400000000000002e-11*exp((-0.59799999999999998)*log(abs(T)))*x4_4 - 4.8987999999999998e-17*exp((0.47999999999999998)*log(abs(T)))
    )) if truthy((x5_4)) else ((
       0
    )))) - vget(X, 2)*vget(X, 4)*((((
       6.7980000000000007e-9*exp((-1.6040000000000001)*log(abs(T)))*x7_4 - 6.7396000000000002e-10*exp((-1.3320000000000001)*log(abs(T))) + 8.1576000000000009e-11*exp((-0.60399999999999998)*log(abs(T)))*x7_4
    )) if truthy((x5_4)) else ((
       0
    )))) - vget(X, 2)*x17_4*(5.1485802679346868*x1_4*exp((1.0)*log(abs(x11_4)))*x14_4 - 0.87659414490283338*x1_4*x13_4*x15_4 - 3.5068370966299316*x21_4) + vget(X, 4)*vget(X, 8)*(2.7400000000000004e-10*x1_4*x11_4*x19_4 - 8.4600000000000008e-10*x21_4) + 4.5700000000000003e-7*vget(X, 1)*vget(X, 10)*x2_4/((T)*(T)))/(vget(X, 0)*x23_4 + vget(X, 1)*x23_4 + vget(X, 10)*x24_4 + vget(X, 11)*x23_4 + vget(X, 12)*x23_4 + vget(X, 13)*x23_4 + vget(X, 2)*x23_4 + vget(X, 3)*x23_4 + vget(X, 4)*x23_4 + vget(X, 5)*x23_4 + vget(X, 6)*x24_4 + vget(X, 7)*x23_4 + vget(X, 8)*x24_4 + vget(X, 9)*x24_4))
    var x0_5 = 2.5950363272655348e-10*exp((-0.75)*log(abs(T)))
    var x1_5 = 7.1999999999999996e-8/sqrt(T)
    var x2_5 = exp(-0.00010729613733905579*T)
    var x3_5 = vget(X, 5)*x2_5
    var x4_5 = exp((0.94999999999999996)*log(abs(T)))
    var x5_5 = 1.3300135414628029e-18*x4_5
    var x6_5 = exp((-0.5)*log(abs(T)))
    var x7_5 = 1.0/T
    var x8_5 = exp(-37.100000000000001*x7_5)
    var x9_5 = T >= 50.0
    var x10_5 = ((((
       2.0000000000000001e-10*exp((0.40200000000000002)*log(abs(T)))*x8_5 - 3.3099999999999998e-17*exp((1.48)*log(abs(T)))
    )) if truthy((x9_5)) else ((
       0
    ))))
    var x11_5 = Log10
    var x12_5 = 1.0/x11_5
    var x13_5 = log(T)
    var x14_5 = x12_5*x13_5
    var x15_5 = exp((-3.0)*log(abs(x11_5)))
    var x16_5 = exp((-2.0)*log(abs(x11_5)))
    var x17_5 = ((x13_5)*(x13_5))
    var x18_5 = exp((-0.12690000000000001*((x13_5)*(x13_5)*(x13_5))*x15_5 - 1.5229999999999999*x14_5 + 1.1180000000000001*x16_5*x17_5 - 19.379999999999999)*log(abs(10.0)))
    var x19_5 = vget(X, 5)*x18_5
    var x20_5 = 6.1739095063118665e-10*exp((0.40999999999999998)*log(abs(T)))
    var x21_5 = exp((-2)*log(abs(T)))
    var x22_5 = 5.25e-11*exp(173900.0*x21_5 - 4430.0*x7_5)
    var x23_5 = T > 200.0
    var x24_5 = ((((
       x22_5
    )) if truthy((x23_5)) else ((
       0
    ))))
    var x25_4 = exp(-33.0*x7_5)
    var x26_4 = ((((
       2.0299999999999998e-9*exp((-0.33200000000000002)*log(abs(T))) + 2.0600000000000001e-10*exp((0.39600000000000002)*log(abs(T)))*x25_4
    )) if truthy((x9_5)) else ((
       0
    ))))
    var x27_4 = 2.6534040307116387e-9*exp((-0.10000000000000001)*log(abs(T)))
    var x28_4 = powi_m5(x11_5)
    var x29_4 = exp((-4)*log(abs(x11_5)))
    var x30_4 = ((((x13_5)*(x13_5)))*(((x13_5)*(x13_5))))
    var x31_4 = powi_m3(x11_5)
    var x32_4 = ((x13_5)*(x13_5)*(x13_5))
    var x33_4 = exp((-2)*log(abs(x11_5)))
    var x34_4 = ((x13_5)*(x13_5))
    var x35_4 = exp((0.31788699999999998*((x13_5)*(x13_5)*(x13_5)*(x13_5)*(x13_5))*x28_4 + 5.8888600000000002*x14_5 - 2.1690299999999998*x29_4*x30_4 + 2.2506900000000001*x31_4*x32_4 + 7.1969200000000004*x33_4*x34_4 - 56.473700000000001)*log(abs(10.0)))
    var x36_4 = T <= 1167.4796423742259
    var x37_4 = exp(-5207.0*x7_5)
    var x38_4 = ((((
       x35_4
    )) if truthy((x36_4)) else ((
       3.1699999999999999e-10*x37_4
    ))))
    var x39_4 = exp((-1.5)*log(abs(T)))*vget(X, 7)
    var x40_4 = exp((-0.59000000000000008)*log(abs(T)))
    var x41_4 = x12_5*x7_5
    var x42_4 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x43_4 = 2.0860422997526066e-16*x42_4
    var x44_3 = 3.4767371836380304e-16*x42_4
    mset(jac, 5, 0, vget(X, 4)*x0_5 + vget(X, 9)*x1_5 - x3_5*x5_5)
    mset(jac, 5, 1, -vget(X, 5)*x10_5 + 7.9674337148168363e-7*vget(X, 7)*x6_5 - x19_5)
    mset(jac, 5, 2, vget(X, 10)*x24_5 + vget(X, 4)*x26_4 - 1.0e-25*vget(X, 5) + vget(X, 7)*x20_5)
    mset(jac, 5, 3, -vget(X, 5)*x20_5 - vget(X, 5)*x27_4)
    mset(jac, 5, 4, vget(X, 0)*x0_5 + vget(X, 2)*x26_4 + 1.9745379206285203e-6*vget(X, 7)*x6_5)
    mset(jac, 5, 5, -vget(X, 0)*x2_5*x5_5 - vget(X, 1)*x10_5 - vget(X, 1)*x18_5 - 1.0e-25*vget(X, 2) - vget(X, 3)*x20_5 - vget(X, 3)*x27_4 - vget(X, 8)*x38_4)
    mset(jac, 5, 6, 0)
    mset(jac, 5, 7, 7.9674337148168363e-7*vget(X, 1)*x6_5 + vget(X, 2)*x20_5 + 1.9745379206285203e-6*vget(X, 4)*x6_5)
    mset(jac, 5, 8, -vget(X, 5)*x38_4)
    mset(jac, 5, 9, vget(X, 0)*x1_5)
    mset(jac, 5, 10, vget(X, 2)*x24_5)
    mset(jac, 5, 11, 0)
    mset(jac, 5, 12, 0)
    mset(jac, 5, 13, 0)
    mset(jac, 5, 14, (-1.9462772454491511e-10*exp((-1.75)*log(abs(T)))*vget(X, 0)*vget(X, 4) + 2.6534040307116389e-10*exp((-1.1000000000000001)*log(abs(T)))*vget(X, 3)*vget(X, 5) - 1.2635128643896626e-18*exp((-0.050000000000000044)*log(abs(T)))*vget(X, 0)*x3_5 + 1.4270531560759686e-22*vget(X, 0)*vget(X, 5)*x2_5*x4_5 - vget(X, 1)*vget(X, 5)*((((
       7.4200000000000004e-9*exp((-1.5979999999999999)*log(abs(T)))*x8_5 + 8.0400000000000002e-11*exp((-0.59799999999999998)*log(abs(T)))*x8_5 - 4.8987999999999998e-17*exp((0.47999999999999998)*log(abs(T)))
    )) if truthy((x9_5)) else ((
       0
    )))) - vget(X, 1)*x19_5*(5.1485802679346868*exp((1.0)*log(abs(x13_5)))*x16_5*x7_5 - 0.87659414490283338*x15_5*x17_5*x7_5 - 3.5068370966299316*x41_4) - 3.9837168574084181e-7*vget(X, 1)*x39_4 + vget(X, 10)*vget(X, 2)*((((
       x22_5*(4430.0*x21_5 - 347800.0/((T)*(T)*(T)))
    )) if truthy((x23_5)) else ((
       0
    )))) + vget(X, 2)*vget(X, 4)*((((
       6.7980000000000007e-9*exp((-1.6040000000000001)*log(abs(T)))*x25_4 - 6.7396000000000002e-10*exp((-1.3320000000000001)*log(abs(T))) + 8.1576000000000009e-11*exp((-0.60399999999999998)*log(abs(T)))*x25_4
    )) if truthy((x9_5)) else ((
       0
    )))) + 2.5313028975878652e-10*vget(X, 2)*vget(X, 7)*x40_4 - 2.5313028975878652e-10*vget(X, 3)*vget(X, 5)*x40_4 - 9.8726896031426014e-7*vget(X, 4)*x39_4 - vget(X, 5)*vget(X, 8)*((((
       x11_5*x35_4*(14.393840000000001*x13_5*x33_4*x7_5 + 1.5894349999999999*x28_4*x30_4*x7_5 - 8.6761199999999992*x29_4*x32_4*x7_5 + 6.7520699999999998*x31_4*x34_4*x7_5 + 5.8888600000000002*x41_4)
    )) if truthy((x36_4)) else ((
       1.650619e-6*x21_5*x37_4
    )))) - 3.5999999999999998e-8*vget(X, 0)*vget(X, 9)/exp((3.0/2.0)*log(abs(T))))/(vget(X, 0)*x43_4 + vget(X, 1)*x43_4 + vget(X, 10)*x44_3 + vget(X, 11)*x43_4 + vget(X, 12)*x43_4 + vget(X, 13)*x43_4 + vget(X, 2)*x43_4 + vget(X, 3)*x43_4 + vget(X, 4)*x43_4 + vget(X, 5)*x43_4 + vget(X, 6)*x44_3 + vget(X, 7)*x43_4 + vget(X, 8)*x44_3 + vget(X, 9)*x44_3))
    var x0_6 = ((T)*(T))
    var x1_6 = ((T)*(T)*(T))
    var x2_6 = ((((T)*(T)))*(((T)*(T))))
    var x3_6 = T <= 10000.0
    var x4_6 = ((((
       -5.5279999999999998e-28*((T)*(T)*(T)*(T)*(T)) - 2.3088e-11*T + 7.3427999999999993e-15*x0_6 - 7.5474000000000004e-19*x1_6 + 3.3467999999999999e-23*x2_6 + 4.2277999999999996e-8
    )) if truthy((x3_6)) else ((
       0
    ))))
    var x5_6 = 1.0e-8*exp((-0.40000000000000002)*log(abs(T)))
    var x6_6 = T < 30
    var x7_6 = log(T)
    var x8_6 = Log10
    var x9_6 = 3.194/x8_6
    var x10_6 = powi_m3(x8_6)
    var x11_6 = ((x7_6)*(x7_6)*(x7_6))
    var x12_6 = exp((-2)*log(abs(x8_6)))
    var x13_6 = ((x7_6)*(x7_6))
    var x14_6 = exp((-0.2072*x10_6*x11_6 + 1.786*x12_6*x13_6 - x7_6*x9_6 - 18.199999999999999)*log(abs(10.0)))
    var x15_6 = ((((
       3.4977396723747635e-20*exp((-0.14999999999999999)*log(abs(T)))
    )) if truthy((x6_6)) else ((
       x14_6
    ))))
    var x16_6 = 1.0/T
    var x17_6 = exp(-21237.150000000001*x16_6)
    var x18_6 = ((((x7_6)*(x7_6)))*(((x7_6)*(x7_6))))
    var x19_6 = ((x7_6)*(x7_6)*(x7_6)*(x7_6)*(x7_6))
    var x20_6 = exp((6)*log(abs(x7_6)))
    var x21_6 = x17_6*(3.4172804999999998e-8*x11_6 - 1.4491368e-7*x13_6 - 4.7813727999999997e-9*x18_6 + 3.9731542e-10*x19_6 - 1.8171411000000001e-11*x20_6 + 3.5311931999999998e-13*((x7_6)*(x7_6)*(x7_6)*(x7_6)*(x7_6)*(x7_6)*(x7_6)) + 3.3735381999999997e-7*x7_6 - 3.3232183000000002e-7)
    var x22_6 = T >= 100.0  and  T <= 30000.0
    var x23_6 = ((((
       x21_6
    )) if truthy((x22_6)) else ((
       0
    ))))
    var x24_6 = 4.9999999999999996e-6/sqrt(T)
    var x25_5 = x13_6*x16_6
    var x26_5 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x27_5 = 2.0860422997526066e-16*x26_5
    var x28_5 = 3.4767371836380304e-16*x26_5
    mset(jac, 6, 0, -vget(X, 6)*x4_6)
    mset(jac, 6, 1, vget(X, 2)*x15_6 + vget(X, 3)*x5_6 + vget(X, 8)*x23_6)
    mset(jac, 6, 2, vget(X, 1)*x15_6 - 6.0e-10*vget(X, 6))
    mset(jac, 6, 3, vget(X, 1)*x5_6 - vget(X, 6)*x24_6)
    mset(jac, 6, 4, 0)
    mset(jac, 6, 5, 0)
    mset(jac, 6, 6, -vget(X, 0)*x4_6 - 6.0e-10*vget(X, 2) - vget(X, 3)*x24_6)
    mset(jac, 6, 7, 0)
    mset(jac, 6, 8, vget(X, 1)*x23_6)
    mset(jac, 6, 9, 0)
    mset(jac, 6, 10, 0)
    mset(jac, 6, 11, 0)
    mset(jac, 6, 12, 0)
    mset(jac, 6, 13, 0)
    mset(jac, 6, 14, (-4.0000000000000002e-9*exp((-1.3999999999999999)*log(abs(T)))*vget(X, 1)*vget(X, 3) - vget(X, 0)*vget(X, 6)*((((
       1.4685599999999999e-14*T - 2.2642200000000001e-18*x0_6 + 1.3387199999999999e-22*x1_6 - 2.7639999999999999e-27*x2_6 - 2.3088e-11
    )) if truthy((x3_6)) else ((
       0
    )))) + vget(X, 1)*vget(X, 2)*((((
       -5.2466095085621454e-21*exp((-1.1499999999999999)*log(abs(T)))
    )) if truthy((x6_6)) else ((
       x14_6*x8_6*(-0.62159999999999993*x10_6*x25_5 + 3.5720000000000001*x12_6*x16_6*x7_6 - x16_6*x9_6)
    )))) + vget(X, 1)*vget(X, 8)*((((
       x17_6*(-1.9125491199999999e-8*x11_6*x16_6 + 1.9865770999999999e-9*x16_6*x18_6 - 1.09028466e-10*x16_6*x19_6 + 2.4718352399999997e-12*x16_6*x20_6 - 2.8982736e-7*x16_6*x7_6 + 3.3735381999999997e-7*x16_6 + 1.0251841499999999e-7*x25_5) + 21237.150000000001*x21_6/x0_6
    )) if truthy((x22_6)) else ((
       0
    )))) + 2.4999999999999998e-6*vget(X, 3)*vget(X, 6)/exp((3.0/2.0)*log(abs(T))))/(vget(X, 0)*x27_5 + vget(X, 1)*x27_5 + vget(X, 10)*x28_5 + vget(X, 11)*x27_5 + vget(X, 12)*x27_5 + vget(X, 13)*x27_5 + vget(X, 2)*x27_5 + vget(X, 3)*x27_5 + vget(X, 4)*x27_5 + vget(X, 5)*x27_5 + vget(X, 6)*x28_5 + vget(X, 7)*x27_5 + vget(X, 8)*x28_5 + vget(X, 9)*x28_5))
    var x0_7 = exp(-0.00010729613733905579*T)
    var x1_7 = vget(X, 5)*x0_7
    var x2_7 = exp((0.94999999999999996)*log(abs(T)))
    var x3_7 = 1.3300135414628029e-18*x2_7
    var x4_7 = exp((-0.5)*log(abs(T)))
    var x5_7 = vget(X, 7)*x4_7
    var x6_7 = 2.6534040307116387e-9*exp((-0.10000000000000001)*log(abs(T)))
    var x7_7 = 6.1739095063118665e-10*exp((0.40999999999999998)*log(abs(T)))
    var x8_7 = exp((-1.5)*log(abs(T)))*vget(X, 7)
    var x9_7 = vget(X, 2)*vget(X, 7)
    var x10_7 = 2.5313028975878652e-10*exp((-0.59000000000000008)*log(abs(T)))
    var x11_7 = vget(X, 0)*x1_7
    var x12_7 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x13_7 = 2.0860422997526066e-16*x12_7
    var x14_7 = 3.4767371836380304e-16*x12_7
    mset(jac, 7, 0, x1_7*x3_7)
    mset(jac, 7, 1, -7.9674337148168363e-7*x5_7)
    mset(jac, 7, 2, -vget(X, 7)*x6_7 - vget(X, 7)*x7_7)
    mset(jac, 7, 3, vget(X, 5)*x7_7)
    mset(jac, 7, 4, -9.8726896031426014e-7*x5_7)
    mset(jac, 7, 5, vget(X, 0)*x0_7*x3_7 + vget(X, 3)*x7_7)
    mset(jac, 7, 6, 0)
    mset(jac, 7, 7, -7.9674337148168363e-7*vget(X, 1)*x4_7 - vget(X, 2)*x6_7 - vget(X, 2)*x7_7 - 9.8726896031426014e-7*vget(X, 4)*x4_7)
    mset(jac, 7, 8, 0)
    mset(jac, 7, 9, 0)
    mset(jac, 7, 10, 0)
    mset(jac, 7, 11, 0)
    mset(jac, 7, 12, 0)
    mset(jac, 7, 13, 0)
    mset(jac, 7, 14, (2.6534040307116389e-10*exp((-1.1000000000000001)*log(abs(T)))*x9_7 + 1.2635128643896626e-18*exp((-0.050000000000000044)*log(abs(T)))*x11_7 + 3.9837168574084181e-7*vget(X, 1)*x8_7 + vget(X, 3)*vget(X, 5)*x10_7 + 4.9363448015713007e-7*vget(X, 4)*x8_7 - x10_7*x9_7 - 1.4270531560759686e-22*x11_7*x2_7)/(vget(X, 0)*x13_7 + vget(X, 1)*x13_7 + vget(X, 10)*x14_7 + vget(X, 11)*x13_7 + vget(X, 12)*x13_7 + vget(X, 13)*x13_7 + vget(X, 2)*x13_7 + vget(X, 3)*x13_7 + vget(X, 4)*x13_7 + vget(X, 5)*x13_7 + vget(X, 6)*x14_7 + vget(X, 7)*x13_7 + vget(X, 8)*x14_7 + vget(X, 9)*x14_7))
    var x0_8 = 1.0/T
    var x1_8 = exp(-46707.0*x0_8)
    var x2_8 = vget(X, 8)*x1_8
    var x3_8 = 35.5*exp((-2.2799999999999998)*log(abs(T)))
    var x4_8 = exp(-102000.0*x0_8)
    var x5_8 = vget(X, 8)*x4_8
    var x6_8 = 4.3799999999999999e-10*exp((0.34999999999999998)*log(abs(T)))
    var x7_8 = exp(-21237.150000000001*x0_8)
    var x8_8 = log(T)
    var x9_8 = ((x8_8)*(x8_8))
    var x10_8 = ((x8_8)*(x8_8)*(x8_8))
    var x11_8 = ((((x8_8)*(x8_8)))*(((x8_8)*(x8_8))))
    var x12_8 = ((x8_8)*(x8_8)*(x8_8)*(x8_8)*(x8_8))
    var x13_8 = exp((6)*log(abs(x8_8)))
    var x14_8 = x7_8*(3.4172804999999998e-8*x10_8 - 4.7813727999999997e-9*x11_8 + 3.9731542e-10*x12_8 - 1.8171411000000001e-11*x13_8 + 3.5311931999999998e-13*((x8_8)*(x8_8)*(x8_8)*(x8_8)*(x8_8)*(x8_8)*(x8_8)) + 3.3735381999999997e-7*x8_8 - 1.4491368e-7*x9_8 - 3.3232183000000002e-7)
    var x15_7 = T >= 100.0  and  T <= 30000.0
    var x16_7 = ((((
       x14_8
    )) if truthy((x15_7)) else ((
       0
    ))))
    var x17_7 = exp(-457.0*x0_8)
    var x18_7 = 1.0000000000000001e-9*x17_7
    var x19_7 = 1.1800000000000001e-10*exp(-69500.0*x0_8)
    var x20_7 = log(x19_7)
    var x21_7 = vget(X, 1) + vget(X, 10) + vget(X, 2) + vget(X, 3) + 2.0*vget(X, 6) + 2.0*vget(X, 8) + vget(X, 9)
    var x22_7 = log(0.0001*T)
    var x23_7 = Log10
    var x24_7 = 1.0/x23_7
    var x25_6 = exp((-2)*log(abs(x23_7)))
    var x26_6 = exp((-1.6200000000000001*((x22_7)*(x22_7))*x25_6 + 1.3*x22_7*x24_7 - 4.8449999999999998)*log(abs(10.0)))
    var x27_6 = x21_7*x26_6
    var x28_6 = x27_6 + 1.0
    var x29_5 = exp((-2)*log(abs(x28_6)))
    var x30_5 = 1.0*x29_5
    var x31_5 = x20_7*x30_5
    var x32_5 = ((vget(X, 8))*(vget(X, 8)))
    var x33_5 = 1.0/x28_6
    var x34_5 = 1.0*x33_5
    var x35_5 = exp((x34_5)*log(abs(x19_7)))
    var x36_5 = sqrt(T)
    var x37_5 = 1.0/x36_5
    var x38_5 = 1.0 - exp(-6000.0*x0_8)
    var x39_5 = 52000.0*x0_8
    var x40_5 = exp(-x39_5)
    var x41_5 = x38_5*x40_5
    var x42_5 = 8.1250000000000003e-8*x37_5*x41_5
    var x43_5 = 1.0 - x34_5
    var x44_4 = exp((x43_5)*log(abs(x42_5)))
    var x45_3 = x35_5*x44_4
    var x46_3 = x32_5*x45_3
    var x47_3 = x26_6*x46_3
    var x48_3 = log(x42_5)
    var x49_3 = x30_5*x48_3
    var x50_3 = x25_6*x9_8
    var x51_3 = -133.82830000000001*x0_8 - 4.8909149999999997*x24_7*x8_8 + 0.47490300000000002*x50_3
    var x52_3 = exp((x51_3 + 14.82123)*log(abs(10.0)))
    var x53_3 = 1.0/x52_3
    var x54_3 = x21_7*x53_3
    var x55_3 = exp(-0.0022727272727272726*T)
    var x56_3 = exp(-0.00054054054054054055*T)
    var x57_3 = -2.0563129999999998*x55_3 + 0.58640729999999996*x56_3 + 0.82274429999999998
    var x58_3 = exp((x57_3)*log(abs(x54_3)))
    var x59_3 = x58_3 + 1.0
    var x60_3 = 1.0/x59_3
    var x61_3 = x24_7*x8_8
    var x62_3 = 16780.950000000001*x0_8 + 1.0
    var x63_3 = powi_m3(x23_7)
    var x64_3 = x10_8*x63_3
    var x65_3 = 40870.379999999997*x0_8 + 1.0
    var x66_3 = -69.700860000000006*x24_7*log(x65_3) + 4.6331670000000003*x64_3
    var x67_3 = 19.734269999999999*x24_7*log(x62_3) + 37.886913*x25_6*x9_8 - 14.509090000000008*x61_3 - x66_3 - 307.31920000000002
    var x68_3 = exp((x51_3 + 13.656822)*log(abs(10.0)))
    var x69_2 = 1.0/x68_3
    var x70_2 = x21_7*x69_2
    var x71_2 = exp((x57_3)*log(abs(x70_2)))
    var x72_2 = x71_2 + 1.0
    var x73_2 = 1.0/x72_2
    var x74_2 = exp((-2080.4099999999999*x0_8*x73_2 - 23705.700000000001*x0_8 + 43.20243*x25_6*x9_8 - x60_3*x67_3 - 68.422430000000006*x61_3 - x66_3 - 178.4239)*log(abs(10.0)))
    var x75_2 = x0_8*x71_2/((x72_2)*(x72_2))
    var x76_2 = 4790.3210533157426*x75_2
    var x77_2 = x68_3*x69_2
    var x78_2 = 1.0/x21_7
    var x79_2 = x57_3*x78_2
    var x80_2 = x77_2*x79_2
    var x81_2 = x58_3*x67_3/((x59_3)*(x59_3))
    var x82_2 = 2.3025850929940459*x81_2
    var x83_2 = x52_3*x53_3
    var x84_2 = x79_2*x83_2
    var x85_2 = 743.05999999999995*x0_8 - 2.4640089999999999*x24_7*x8_8 + 0.19859550000000001*x50_3
    var x86_2 = exp((x85_2 + 9.3055640000000004)*log(abs(10.0)))
    var x87_2 = 1.0/x86_2
    var x88_2 = x21_7*x87_2
    var x89_1 = 2.9375070000000001*x55_3 + 0.23588480000000001*x56_3 + 0.75022860000000002
    var x90_1 = exp((x89_1)*log(abs(x88_2)))
    var x91_1 = x90_1 + 1.0
    var x92_1 = 1.0/x91_1
    var x93_1 = 14254.549999999999*x0_8 + 1.0
    var x94_1 = 27535.310000000001*x0_8 + 1.0
    var x95_1 = -21.360939999999999*x24_7*log(x94_1) + 0.25820969999999999*x64_3
    var x96_1 = 70.138370000000009*x24_7*x8_8 + 11.28215*x24_7*log(x93_1) - 4.7035149999999994*x50_3 - x95_1 - 203.11568
    var x97_1 = exp((x85_2 + 8.1313220000000008)*log(abs(10.0)))
    var x98_1 = 1.0/x97_1
    var x99_1 = x21_7*x98_1
    var x100_1 = exp((x89_1)*log(abs(x99_1)))
    var x101_1 = x100_1 + 1.0
    var x102_1 = 1.0/x101_1
    var x103_1 = exp((-1657.4099999999999*x0_8*x102_1 - 21467.790000000001*x0_8 + 42.707410000000003*x24_7*x8_8 - 2.0273650000000001*x50_3 - x92_1*x96_1 - x95_1 - 142.7664)*log(abs(10.0)))
    var x104_1 = x0_8*x100_1/((x101_1)*(x101_1))
    var x105_1 = 3816.3275589792611*x104_1
    var x106_1 = x97_1*x98_1
    var x107_1 = x78_2*x89_1
    var x108_1 = x106_1*x107_1
    var x109_1 = x90_1*x96_1/((x91_1)*(x91_1))
    var x110_1 = 2.3025850929940459*x109_1
    var x111_1 = x86_2*x87_2
    var x112_1 = x107_1*x111_1
    var x113_1 = vget(X, 2)*vget(X, 8)
    var x114_1 = x113_1*(-x103_1*(x105_1*x108_1 + x110_1*x112_1) - x74_2*(x76_2*x80_2 + x82_2*x84_2)) + x31_5*x47_3 - x47_3*x49_3
    var x115_1 = exp((-2)*log(abs(T)))
    var x116_1 = 5.25e-11*exp(-4430.0*x0_8 + 173900.0*x115_1)
    var x117_1 = T > 200.0
    var x118_1 = ((((
       x116_1
    )) if truthy((x117_1)) else ((
       0
    ))))
    var x119_1 = -x103_1 - x74_2
    var x120_1 = ((vget(X, 2))*(vget(X, 2)))
    var x121_1 = exp((-0.25)*log(abs(T)))
    var x122_1 = 0.0061910000000000003*exp((1.0461)*log(abs(T))) + 8.9711999999999997e-11*exp((3.0424000000000002)*log(abs(T))) + 3.2575999999999999e-14*exp((3.7740999999999998)*log(abs(T))) + 1.0
    var x123_1 = 1.0/x122_1
    var x124_1 = 1.3500000000000001e-9*exp((0.098492999999999997)*log(abs(T))) + 4.4350199999999998e-10*exp((0.55610000000000004)*log(abs(T))) + 3.7408500000000004e-16*exp((2.1825999999999999)*log(abs(T)))
    var x125_1 = x123_1*x124_1
    var x126_1 = 1.5e-32*x121_1 + 5.0000000000000004e-32*x37_5
    var x127_1 = 2*x113_1
    var x128_1 = -7.5000000000000001e-33*x121_1 - 2.5000000000000002e-32*x37_5
    var x129_1 = 4.9999999999999996e-6/sqrt(T)
    var x130_1 = 1.3700000000000002e-10*x25_6*x9_8 - 8.4600000000000008e-10*x61_3 - 4.1700000000000001e-10
    var x131_1 = powi_m5(x23_7)
    var x132_1 = exp((-4)*log(abs(x23_7)))
    var x133_1 = exp((-2.1690299999999998*x11_8*x132_1 + 0.31788699999999998*x12_8*x131_1 + 7.1969200000000004*x50_3 + 5.8888600000000002*x61_3 + 2.2506900000000001*x64_3 - 56.473700000000001)*log(abs(10.0)))
    var x134_1 = T <= 1167.4796423742259
    var x135_1 = exp(-5207.0*x0_8)
    var x136_1 = ((((
       x133_1
    )) if truthy((x134_1)) else ((
       3.1699999999999999e-10*x135_1
    ))))
    var x137_1 = 2.0*x29_5*x47_3
    var x138_1 = x137_1*x48_3
    var x139_1 = -x103_1*(7632.6551179585222*x104_1*x108_1 + 4.6051701859880918*x109_1*x112_1) - x74_2*(9580.6421066314851*x75_2*x80_2 + 4.6051701859880918*x81_2*x84_2)
    var x140_1 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x141_1 = 2.0860422997526066e-16*x140_1
    var x142_1 = 3.4767371836380304e-16*x140_1
    var x143_1 = vget(X, 0)*x2_8
    var x144_1 = vget(X, 0)*x5_8
    var x145_1 = exp((-1.5)*log(abs(T)))
    var x146_1 = exp((-1.25)*log(abs(T)))
    var x147_1 = vget(X, 8)*x120_1
    var x148_1 = x0_8*x24_7
    var x149_1 = x0_8*x8_8
    var x150_1 = x149_1*x25_6
    var x151_1 = vget(X, 2)*vget(X, 3)
    var x152_1 = x0_8*x9_8
    var x153_1 = x0_8*x10_8
    var x154_1 = x0_8*x11_8
    var x155_1 = x152_1*x63_3
    var x156_1 = x27_6*(-7.460375701300709*x0_8*x22_7*x25_6 + 2.9933606208922598*x0_8*x24_7)
    var x157_1 = exp((-2.5)*log(abs(T)))
    var x158_1 = x115_1*x24_7
    var x159_1 = x158_1/x65_3
    var x160_1 = 0.0046734386363636356*x55_3 - 0.00031697691891891889*x56_3
    var x161_1 = x57_3*(11.261747970100974*x0_8*x24_7 - 308.15104860073512*x115_1 - 2.1870091368363029*x150_1)
    var x162_1 = x158_1/x94_1
    var x163_1 = -0.0066761522727272725*x55_3 - 0.0001275052972972973*x56_3
    var x164_1 = x89_1*(1710.9588792001557*x115_1 + 5.6735903924031659*x148_1 - 0.91456607567139814*x150_1)
    mset(jac, 8, 0, -x2_8*x3_8 - x5_8*x6_8)
    mset(jac, 8, 1, vget(X, 10)*x18_7 - vget(X, 8)*x16_7 + x114_1)
    mset(jac, 8, 2, vget(X, 10)*x118_1 + vget(X, 3)*x125_1 + 6.0e-10*vget(X, 6) + vget(X, 8)*x119_1 + x114_1 + 3*x120_1*(6.0000000000000001e-32*x121_1 + 2.0000000000000002e-31*x37_5) + x126_1*x127_1 + x127_1*x128_1)
    mset(jac, 8, 3, vget(X, 2)*x125_1 + vget(X, 6)*x129_1 + x114_1)
    mset(jac, 8, 4, vget(X, 8)*x130_1)
    mset(jac, 8, 5, -vget(X, 8)*x136_1)
    mset(jac, 8, 6, 6.0e-10*vget(X, 2) + vget(X, 3)*x129_1 + x113_1*x139_1 + x137_1*x20_7 - x138_1)
    mset(jac, 8, 7, 0)
    mset(jac, 8, 8, -vget(X, 0)*x1_8*x3_8 - vget(X, 0)*x4_8*x6_8 - vget(X, 1)*x16_7 + vget(X, 2)*vget(X, 8)*x139_1 + vget(X, 2)*x119_1 + vget(X, 4)*x130_1 - vget(X, 5)*x136_1 - 2*vget(X, 8)*x45_3 + x120_1*x126_1 + x120_1*x128_1 - x138_1 + 2.0*x20_7*x26_6*x29_5*x32_5*x35_5*x44_4)
    mset(jac, 8, 9, x114_1)
    mset(jac, 8, 10, vget(X, 1)*x18_7 + vget(X, 2)*x118_1 + x114_1)
    mset(jac, 8, 11, 0)
    mset(jac, 8, 12, 0)
    mset(jac, 8, 13, 0)
    mset(jac, 8, 14, (-1658098.5*exp((-4.2799999999999994)*log(abs(T)))*x143_1 + 80.939999999999998*exp((-3.2799999999999998)*log(abs(T)))*x143_1 - 4.4675999999999997e-5*exp((-1.6499999999999999)*log(abs(T)))*x144_1 - 1.5329999999999998e-10*exp((-0.65000000000000002)*log(abs(T)))*x144_1 + 4.5700000000000003e-7*vget(X, 1)*vget(X, 10)*x115_1*x17_7 - vget(X, 1)*vget(X, 8)*((((
       21237.150000000001*x115_1*x14_8 + x7_8*(-1.09028466e-10*x0_8*x12_8 + 2.4718352399999997e-12*x0_8*x13_8 + 3.3735381999999997e-7*x0_8 - 2.8982736e-7*x149_1 + 1.0251841499999999e-7*x152_1 - 1.9125491199999999e-8*x153_1 + 1.9865770999999999e-9*x154_1)
    )) if truthy((x15_7)) else ((
       0
    )))) + vget(X, 10)*vget(X, 2)*((((
       x116_1*(4430.0*x115_1 - 347800.0/((T)*(T)*(T)))
    )) if truthy((x117_1)) else ((
       0
    )))) + ((vget(X, 2))*(vget(X, 2))*(vget(X, 2)))*(-1.0000000000000001e-31*x145_1 - 1.5e-32*x146_1) + vget(X, 4)*vget(X, 8)*(-8.4600000000000008e-10*x148_1 + 2.7400000000000004e-10*x150_1) - vget(X, 5)*vget(X, 8)*((((
       x133_1*x23_7*(1.5894349999999999*x131_1*x154_1 - 8.6761199999999992*x132_1*x153_1 + 5.8888600000000002*x148_1 + 14.393840000000001*x150_1 + 6.7520699999999998*x155_1)
    )) if truthy((x134_1)) else ((
       1.650619e-6*x115_1*x135_1
    )))) + x113_1*(-x103_1*(3816.3275589792611*x102_1*x115_1 + x105_1*(x106_1*x164_1 + x163_1*log(x99_1)) + x110_1*(x111_1*x164_1 + x163_1*log(x88_2)) + 49431.413233526648*x115_1 + 98.337445626384849*x148_1 - 9.3363608541157479*x150_1 - 1.783649418259394*x155_1 - 1354334.7412883535*x162_1 - 2.3025850929940459*x92_1*(70.138370000000009*x0_8*x24_7 - 9.4070299999999989*x150_1 - 0.77462909999999996*x155_1 - 160821.97128249999*x158_1/x93_1 - 588180.10479140002*x162_1)) - x74_2*(4790.3210533157426*x115_1*x73_2 + 54584.391438988954*x115_1 - 157.54846734442862*x148_1 + 198.95454259823751*x150_1 - 32.004783802655837*x155_1 - 6559375.6154640894*x159_1 - 2.3025850929940459*x60_3*(75.773826*x0_8*x25_6*x8_8 - 14.509090000000008*x148_1 - 13.899501000000001*x155_1 - 331159.79815649998*x158_1/x62_3 - 2848700.6345267999*x159_1) + x76_2*(x160_1*log(x70_2) + x161_1*x77_2) + x82_2*(x160_1*log(x54_3) + x161_1*x83_2))) + x123_1*x151_1*(1.3296555000000001e-10*exp((-0.90150700000000006)*log(abs(T))) + 2.466314622e-10*exp((-0.44389999999999996)*log(abs(T))) + 8.1647792100000001e-16*exp((1.1825999999999999)*log(abs(T)))) + x147_1*(-2.5000000000000002e-32*x145_1 - 3.75e-33*x146_1) + x147_1*(1.2500000000000001e-32*x145_1 + 1.875e-33*x146_1) - x46_3*(69500.0*x115_1*x33_5 - x156_1*x31_5) - x46_3*(x156_1*x49_3 + 12307692.307692308*x36_5*x43_5*(-4.0625000000000001e-8*x145_1*x41_5 + 0.0042250000000000005*x157_1*x38_5*x40_5 - 0.00048750000000000003*x157_1*exp(-58000.0*x0_8))*exp(x39_5)/x38_5) + x124_1*x151_1*(-0.0064764051000000007*exp((0.04610000000000003)*log(abs(T))) - 2.7293978880000002e-10*exp((2.0424000000000002)*log(abs(T))) - 1.229450816e-13*exp((2.7740999999999998)*log(abs(T))))/((x122_1)*(x122_1)) - 2.4999999999999998e-6*vget(X, 3)*vget(X, 6)/exp((3.0/2.0)*log(abs(T))))/(vget(X, 0)*x141_1 + vget(X, 1)*x141_1 + vget(X, 10)*x142_1 + vget(X, 11)*x141_1 + vget(X, 12)*x141_1 + vget(X, 13)*x141_1 + vget(X, 2)*x141_1 + vget(X, 3)*x141_1 + vget(X, 4)*x141_1 + vget(X, 5)*x141_1 + vget(X, 6)*x142_1 + vget(X, 7)*x141_1 + vget(X, 8)*x142_1 + vget(X, 9)*x142_1))
    var x0_9 = 7.1999999999999996e-8/sqrt(T)
    var x1_9 = log(T)
    var x2_9 = Log10
    var x3_9 = 1.0/x2_9
    var x4_9 = exp((-3.0)*log(abs(x2_9)))
    var x5_9 = exp((-2.0)*log(abs(x2_9)))
    var x6_9 = ((x1_9)*(x1_9))
    var x7_9 = exp((-1.5229999999999999*x1_9*x3_9 - 0.12690000000000001*((x1_9)*(x1_9)*(x1_9))*x4_9 + 1.1180000000000001*x5_9*x6_9 - 19.379999999999999)*log(abs(10.0)))
    var x8_9 = vget(X, 5)*x7_9
    var x9_9 = vget(X, 4)*x7_9
    var x10_9 = 1.0/T
    var x11_9 = 5.1485802679346868*exp((1.0)*log(abs(x1_9)))*x10_9*x5_9 - 3.5068370966299316*x10_9*x3_9 - 0.87659414490283338*x10_9*x4_9*x6_9
    var x12_9 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x13_9 = 2.0860422997526066e-16*x12_9
    var x14_9 = 3.4767371836380304e-16*x12_9
    mset(jac, 9, 0, -vget(X, 9)*x0_9)
    mset(jac, 9, 1, x8_9)
    mset(jac, 9, 2, -6.3999999999999996e-10*vget(X, 9) + x9_9)
    mset(jac, 9, 3, 0)
    mset(jac, 9, 4, vget(X, 2)*x7_9)
    mset(jac, 9, 5, vget(X, 1)*x7_9)
    mset(jac, 9, 6, 0)
    mset(jac, 9, 7, 0)
    mset(jac, 9, 8, 0)
    mset(jac, 9, 9, -vget(X, 0)*x0_9 - 6.3999999999999996e-10*vget(X, 2))
    mset(jac, 9, 10, 0)
    mset(jac, 9, 11, 0)
    mset(jac, 9, 12, 0)
    mset(jac, 9, 13, 0)
    mset(jac, 9, 14, (vget(X, 1)*x11_9*x8_9 + vget(X, 2)*x11_9*x9_9 + 3.5999999999999998e-8*vget(X, 0)*vget(X, 9)/exp((3.0/2.0)*log(abs(T))))/(vget(X, 0)*x13_9 + vget(X, 1)*x13_9 + vget(X, 10)*x14_9 + vget(X, 11)*x13_9 + vget(X, 12)*x13_9 + vget(X, 13)*x13_9 + vget(X, 2)*x13_9 + vget(X, 3)*x13_9 + vget(X, 4)*x13_9 + vget(X, 5)*x13_9 + vget(X, 6)*x14_9 + vget(X, 7)*x13_9 + vget(X, 8)*x14_9 + vget(X, 9)*x14_9))
    var x0_10 = 1.0/T
    var x1_10 = exp(-457.0*x0_10)
    var x2_10 = 1.0000000000000001e-9*x1_10
    var x3_10 = 2.6534040307116387e-9*exp((-0.10000000000000001)*log(abs(T)))
    var x4_10 = exp((-2)*log(abs(T)))
    var x5_10 = 5.25e-11*exp(-4430.0*x0_10 + 173900.0*x4_10)
    var x6_10 = T > 200.0
    var x7_10 = ((((
       x5_10
    )) if truthy((x6_10)) else ((
       0
    ))))
    var x8_10 = Log10
    var x9_10 = 1.0/x8_10
    var x10_10 = log(T)
    var x11_10 = x10_10*x9_10
    var x12_10 = exp((-2)*log(abs(x8_10)))
    var x13_10 = ((x10_10)*(x10_10))
    var x14_10 = x12_10*x13_10
    var x15_8 = 8.4600000000000008e-10*x11_10 - 1.3700000000000002e-10*x14_10 + 4.1700000000000001e-10
    var x16_8 = powi_m5(x8_10)
    var x17_8 = exp((-4)*log(abs(x8_10)))
    var x18_8 = ((((x10_10)*(x10_10)))*(((x10_10)*(x10_10))))
    var x19_8 = powi_m3(x8_10)
    var x20_8 = ((x10_10)*(x10_10)*(x10_10))
    var x21_8 = exp((0.31788699999999998*((x10_10)*(x10_10)*(x10_10)*(x10_10)*(x10_10))*x16_8 + 5.8888600000000002*x11_10 + 7.1969200000000004*x14_10 - 2.1690299999999998*x17_8*x18_8 + 2.2506900000000001*x19_8*x20_8 - 56.473700000000001)*log(abs(10.0)))
    var x22_8 = T <= 1167.4796423742259
    var x23_8 = exp(-5207.0*x0_10)
    var x24_8 = ((((
       x21_8
    )) if truthy((x22_8)) else ((
       3.1699999999999999e-10*x23_8
    ))))
    var x25_7 = 2.6534040307116389e-10*exp((-1.1000000000000001)*log(abs(T)))
    var x26_7 = x0_10*x10_10*x12_10
    var x27_7 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x28_7 = 2.0860422997526066e-16*x27_7
    var x29_6 = 3.4767371836380304e-16*x27_7
    mset(jac, 10, 0, 0)
    mset(jac, 10, 1, -vget(X, 10)*x2_10)
    mset(jac, 10, 2, -vget(X, 10)*x7_10 + 1.0e-25*vget(X, 5) + vget(X, 7)*x3_10 + 6.3999999999999996e-10*vget(X, 9))
    mset(jac, 10, 3, vget(X, 5)*x3_10)
    mset(jac, 10, 4, vget(X, 8)*x15_8)
    mset(jac, 10, 5, 1.0e-25*vget(X, 2) + vget(X, 3)*x3_10 + vget(X, 8)*x24_8)
    mset(jac, 10, 6, 0)
    mset(jac, 10, 7, vget(X, 2)*x3_10)
    mset(jac, 10, 8, vget(X, 4)*x15_8 + vget(X, 5)*x24_8)
    mset(jac, 10, 9, 6.3999999999999996e-10*vget(X, 2))
    mset(jac, 10, 10, -vget(X, 1)*x2_10 - vget(X, 2)*x7_10)
    mset(jac, 10, 11, 0)
    mset(jac, 10, 12, 0)
    mset(jac, 10, 13, 0)
    mset(jac, 10, 14, (-4.5700000000000003e-7*vget(X, 1)*vget(X, 10)*x1_10*x4_10 - vget(X, 10)*vget(X, 2)*((((
       x5_10*(4430.0*x4_10 - 347800.0/((T)*(T)*(T)))
    )) if truthy((x6_10)) else ((
       0
    )))) - vget(X, 2)*vget(X, 7)*x25_7 - vget(X, 3)*vget(X, 5)*x25_7 + vget(X, 4)*vget(X, 8)*(8.4600000000000008e-10*x0_10*x9_10 - 2.7400000000000004e-10*x26_7) + vget(X, 5)*vget(X, 8)*((((
       x21_8*x8_10*(6.7520699999999998*x0_10*x13_10*x19_8 + 1.5894349999999999*x0_10*x16_8*x18_8 - 8.6761199999999992*x0_10*x17_8*x20_8 + 5.8888600000000002*x0_10*x9_10 + 14.393840000000001*x26_7)
    )) if truthy((x22_8)) else ((
       1.650619e-6*x23_8*x4_10
    )))))/(vget(X, 0)*x28_7 + vget(X, 1)*x28_7 + vget(X, 10)*x29_6 + vget(X, 11)*x28_7 + vget(X, 12)*x28_7 + vget(X, 13)*x28_7 + vget(X, 2)*x28_7 + vget(X, 3)*x28_7 + vget(X, 4)*x28_7 + vget(X, 5)*x28_7 + vget(X, 6)*x29_6 + vget(X, 7)*x28_7 + vget(X, 8)*x29_6 + vget(X, 9)*x29_6))
    var x0_11 = sqrt(T)
    var x1_11 = 0.00060040841663220993*x0_11 + 1.0
    var x2_11 = exp((-1.7524)*log(abs(x1_11)))
    var x3_11 = 0.32668576019240059*x0_11 + 1.0
    var x4_11 = exp((-0.24759999999999999)*log(abs(x3_11)))
    var x5_11 = vget(X, 11)*x4_11
    var x6_11 = x2_11*x5_11
    var x7_11 = 5.7884371785482823e-10/x0_11
    var x8_11 = log(8.6173430000000006e-5*T)
    var x9_11 = ((x8_11)*(x8_11))
    var x10_11 = ((x8_11)*(x8_11)*(x8_11))
    var x11_11 = ((((x8_11)*(x8_11)))*(((x8_11)*(x8_11))))
    var x12_11 = ((x8_11)*(x8_11)*(x8_11)*(x8_11)*(x8_11))
    var x13_11 = exp((6)*log(abs(x8_11)))
    var x14_11 = ((x8_11)*(x8_11)*(x8_11)*(x8_11)*(x8_11)*(x8_11)*(x8_11))
    var x15_9 = exp(4.7016264867590021*x10_11 - 0.76924663344919997*x11_11 + 0.081130420973029999*x12_11 - 0.005324020628287001*x13_11 + 0.00019757053122209999*x14_11 - 3.1655810656650001e-6*exp((8)*log(abs(x8_11))) - 18.480669935680002*x9_11)
    var x16_9 = vget(X, 12)*x15_9
    var x17_9 = 3.8571873359681582e-209*exp((43.933476326349997)*log(abs(T)))
    var x18_9 = x16_9*x17_9
    var x19_9 = vget(X, 0)*x2_11
    var x20_9 = 1.0/T
    var x21_9 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x22_9 = 2.0860422997526066e-16*x21_9
    var x23_9 = 3.4767371836380304e-16*x21_9
    mset(jac, 11, 0, x18_9 - x6_11*x7_11)
    mset(jac, 11, 1, 0)
    mset(jac, 11, 2, 0)
    mset(jac, 11, 3, 0)
    mset(jac, 11, 4, 0)
    mset(jac, 11, 5, 0)
    mset(jac, 11, 6, 0)
    mset(jac, 11, 7, 0)
    mset(jac, 11, 8, 0)
    mset(jac, 11, 9, 0)
    mset(jac, 11, 10, 0)
    mset(jac, 11, 11, -x19_9*x4_11*x7_11)
    mset(jac, 11, 12, vget(X, 0)*x15_9*x17_9)
    mset(jac, 11, 13, 0)
    mset(jac, 11, 14, (1.694596485110541e-207*exp((42.933476326349997)*log(abs(T)))*vget(X, 0)*x16_9 + 3.0451686126851684e-13*vget(X, 0)*exp((-2.7523999999999997)*log(abs(x1_11)))*x20_9*x5_11 + vget(X, 0)*x18_9*(-3.0769865337967999*x10_11*x20_9 + 0.40565210486515002*x11_11*x20_9 - 0.031944123769722006*x12_11*x20_9 + 0.0013829937185547*x13_11*x20_9 - 2.5324648525320001e-5*x14_11*x20_9 - 36.961339871360003*x20_9*x8_11 + 14.104879460277006*x20_9*x9_11) + 2.3410580000000002e-11*vget(X, 11)*x19_9*x20_9*exp((-1.2476)*log(abs(x3_11))) + 2.8942185892741411e-10*vget(X, 0)*x6_11/exp((3.0/2.0)*log(abs(T))))/(vget(X, 0)*x22_9 + vget(X, 1)*x22_9 + vget(X, 10)*x23_9 + vget(X, 11)*x22_9 + vget(X, 12)*x22_9 + vget(X, 13)*x22_9 + vget(X, 2)*x22_9 + vget(X, 3)*x22_9 + vget(X, 4)*x22_9 + vget(X, 5)*x22_9 + vget(X, 6)*x23_9 + vget(X, 7)*x22_9 + vget(X, 8)*x23_9 + vget(X, 9)*x23_9))
    var x0_12 = sqrt(T)
    var x1_12 = 0.00060040841663220993*x0_12 + 1.0
    var x2_12 = exp((-1.7524)*log(abs(x1_12)))
    var x3_12 = 0.32668576019240059*x0_12 + 1.0
    var x4_12 = exp((-0.24759999999999999)*log(abs(x3_12)))
    var x5_12 = vget(X, 11)*x4_12
    var x6_12 = x2_12*x5_12
    var x7_12 = 5.7884371785482823e-10/x0_12
    var x8_12 = 1.4981088130721367e-10*exp((-0.63529999999999998)*log(abs(T)))
    var x9_12 = 8.6173430000000006e-5*T
    var x10_12 = x9_12 <= 9280.0
    var x11_12 = 1.0/T
    var x12_12 = (1.5400000000000001e-9 + 4.6200000000000001e-10*exp(-93988.701501924661*x11_12))*exp(-469943.50750964211*x11_12)
    var x13_12 = ((((
       x8_12
    )) if truthy((x10_12)) else ((
       1250086.112245841*exp((-1.5)*log(abs(T)))*x12_12 + x8_12
    ))))
    var x14_12 = log(x9_12)
    var x15_10 = ((x14_12)*(x14_12))
    var x16_10 = ((x14_12)*(x14_12)*(x14_12))
    var x17_10 = ((((x14_12)*(x14_12)))*(((x14_12)*(x14_12))))
    var x18_10 = ((x14_12)*(x14_12)*(x14_12)*(x14_12)*(x14_12))
    var x19_10 = exp((6)*log(abs(x14_12)))
    var x20_10 = ((x14_12)*(x14_12)*(x14_12)*(x14_12)*(x14_12)*(x14_12)*(x14_12))
    var x21_10 = exp((8)*log(abs(x14_12)))
    var x22_10 = exp(-10.753230200000001*x15_10 + 3.0580387500000001*x16_10 - 0.56851189000000002*x17_10 + 0.067953912300000002*x18_10 - 0.0050090561*x19_10 + 0.000206723616*x20_10 - 3.6491614100000001e-6*x21_10)
    var x23_10 = exp((23.915965629999999)*log(abs(T)))
    var x24_9 = 4.3524079114767552e-117*x23_10
    var x25_8 = exp(-18.480669935680002*x15_10 + 4.7016264867590021*x16_10 - 0.76924663344919997*x17_10 + 0.081130420973029999*x18_10 - 0.005324020628287001*x19_10 + 0.00019757053122209999*x20_10 - 3.1655810656650001e-6*x21_10)
    var x26_8 = 3.8571873359681582e-209*exp((43.933476326349997)*log(abs(T)))*x25_8
    var x27_8 = exp((-0.75)*log(abs(T)))
    var x28_8 = exp(-127500.0*x11_12)
    var x29_7 = T <= 10000.0
    var x30_6 = ((((
       1.26e-9*x27_8*x28_8
    )) if truthy((x29_7)) else ((
       4.0000000000000003e-37*exp((4.7400000000000002)*log(abs(T)))
    ))))
    var x31_6 = 2.8833736969617052e-16*exp((0.25)*log(abs(T)))
    var x32_6 = vget(X, 0)*x2_12
    var x33_6 = vget(X, 0)*vget(X, 12)
    var x34_6 = -9.5174852894472843e-11*exp((-1.6353)*log(abs(T)))
    var x35_6 = exp((-3.5)*log(abs(T)))
    var x36_6 = x11_12*x14_12
    var x37_6 = x11_12*x16_10
    var x38_6 = x11_12*x18_10
    var x39_6 = x11_12*x20_10
    var x40_6 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x41_6 = 2.0860422997526066e-16*x40_6
    var x42_6 = 3.4767371836380304e-16*x40_6
    mset(jac, 12, 0, -vget(X, 12)*x13_12 - vget(X, 12)*x26_8 + vget(X, 13)*x22_10*x24_9 + x6_12*x7_12)
    mset(jac, 12, 1, vget(X, 13)*x30_6)
    mset(jac, 12, 2, -vget(X, 12)*x31_6)
    mset(jac, 12, 3, 0)
    mset(jac, 12, 4, 0)
    mset(jac, 12, 5, 0)
    mset(jac, 12, 6, 0)
    mset(jac, 12, 7, 0)
    mset(jac, 12, 8, 0)
    mset(jac, 12, 9, 0)
    mset(jac, 12, 10, 0)
    mset(jac, 12, 11, x32_6*x4_12*x7_12)
    mset(jac, 12, 12, -vget(X, 0)*x13_12 - vget(X, 0)*x26_8 - vget(X, 2)*x31_6)
    mset(jac, 12, 13, vget(X, 0)*x22_10*x24_9 + vget(X, 1)*x30_6)
    mset(jac, 12, 14, (1.0409203801861816e-115*exp((22.915965629999999)*log(abs(T)))*vget(X, 0)*vget(X, 13)*x22_10 - 1.694596485110541e-207*exp((42.933476326349997)*log(abs(T)))*x25_8*x33_6 + 4.3524079114767552e-117*vget(X, 0)*vget(X, 13)*x22_10*x23_10*(9.1741162500000009*x11_12*x15_10 + 0.33976956150000004*x11_12*x17_10 + 0.001447065312*x11_12*x19_10 - 21.506460400000002*x36_6 - 2.2740475600000001*x37_6 - 0.030054336600000002*x38_6 - 2.9193291280000001e-5*x39_6) - 3.0451686126851684e-13*vget(X, 0)*exp((-2.7523999999999997)*log(abs(x1_12)))*x11_12*x5_12 + vget(X, 1)*vget(X, 13)*((((
       0.00016065*exp((-2.75)*log(abs(T)))*x28_8 - 9.4499999999999994e-10*exp((-1.75)*log(abs(T)))*x28_8
    )) if truthy((x29_7)) else ((
       1.8960000000000001e-36*exp((3.7400000000000002)*log(abs(T)))
    )))) - 2.3410580000000002e-11*vget(X, 11)*x11_12*exp((-1.2476)*log(abs(x3_12)))*x32_6 - 7.2084342424042629e-17*vget(X, 12)*vget(X, 2)*x27_8 - x26_8*x33_6*(14.104879460277006*x11_12*x15_10 + 0.40565210486515002*x11_12*x17_10 + 0.0013829937185547*x11_12*x19_10 - 36.961339871360003*x36_6 - 3.0769865337967999*x37_6 - 0.031944123769722006*x38_6 - 2.5324648525320001e-5*x39_6) - x33_6*((((
       x34_6
    )) if truthy((x10_12)) else ((
       -1875129.1683687614*exp((-2.5)*log(abs(T)))*x12_12 + 587469852277.90271*x12_12*x35_6 + x34_6 + 54.282214350476039*x35_6*exp(-563932.20901156683*x11_12)
    )))) - 2.8942185892741411e-10*vget(X, 0)*x6_12/exp((3.0/2.0)*log(abs(T))))/(vget(X, 0)*x41_6 + vget(X, 1)*x41_6 + vget(X, 10)*x42_6 + vget(X, 11)*x41_6 + vget(X, 12)*x41_6 + vget(X, 13)*x41_6 + vget(X, 2)*x41_6 + vget(X, 3)*x41_6 + vget(X, 4)*x41_6 + vget(X, 5)*x41_6 + vget(X, 6)*x42_6 + vget(X, 7)*x41_6 + vget(X, 8)*x42_6 + vget(X, 9)*x42_6))
    var x0_13 = 1.4981088130721367e-10*exp((-0.63529999999999998)*log(abs(T)))
    var x1_13 = 8.6173430000000006e-5*T
    var x2_13 = x1_13 <= 9280.0
    var x3_13 = 1.0/T
    var x4_13 = (1.5400000000000001e-9 + 4.6200000000000001e-10*exp(-93988.701501924661*x3_13))*exp(-469943.50750964211*x3_13)
    var x5_13 = ((((
       x0_13
    )) if truthy((x2_13)) else ((
       1250086.112245841*exp((-1.5)*log(abs(T)))*x4_13 + x0_13
    ))))
    var x6_13 = log(x1_13)
    var x7_13 = ((x6_13)*(x6_13))
    var x8_13 = ((x6_13)*(x6_13)*(x6_13))
    var x9_13 = ((((x6_13)*(x6_13)))*(((x6_13)*(x6_13))))
    var x10_13 = ((x6_13)*(x6_13)*(x6_13)*(x6_13)*(x6_13))
    var x11_13 = exp((6)*log(abs(x6_13)))
    var x12_13 = ((x6_13)*(x6_13)*(x6_13)*(x6_13)*(x6_13)*(x6_13)*(x6_13))
    var x13_13 = exp(0.067953912300000002*x10_13 - 0.0050090561*x11_13 + 0.000206723616*x12_13 - 3.6491614100000001e-6*exp((8)*log(abs(x6_13))) - 10.753230200000001*x7_13 + 3.0580387500000001*x8_13 - 0.56851189000000002*x9_13)
    var x14_13 = vget(X, 13)*x13_13
    var x15_11 = 4.3524079114767552e-117*exp((23.915965629999999)*log(abs(T)))
    var x16_11 = x14_13*x15_11
    var x17_11 = exp((-0.75)*log(abs(T)))
    var x18_11 = exp(-127500.0*x3_13)
    var x19_11 = T <= 10000.0
    var x20_11 = ((((
       1.26e-9*x17_11*x18_11
    )) if truthy((x19_11)) else ((
       4.0000000000000003e-37*exp((4.7400000000000002)*log(abs(T)))
    ))))
    var x21_11 = 2.8833736969617052e-16*exp((0.25)*log(abs(T)))
    var x22_11 = -9.5174852894472843e-11*exp((-1.6353)*log(abs(T)))
    var x23_11 = exp((-3.5)*log(abs(T)))
    var x24_10 = 1.0/(9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9))
    var x25_9 = 2.0860422997526066e-16*x24_10
    var x26_9 = 3.4767371836380304e-16*x24_10
    mset(jac, 13, 0, vget(X, 12)*x5_13 - x16_11)
    mset(jac, 13, 1, -vget(X, 13)*x20_11)
    mset(jac, 13, 2, vget(X, 12)*x21_11)
    mset(jac, 13, 3, 0)
    mset(jac, 13, 4, 0)
    mset(jac, 13, 5, 0)
    mset(jac, 13, 6, 0)
    mset(jac, 13, 7, 0)
    mset(jac, 13, 8, 0)
    mset(jac, 13, 9, 0)
    mset(jac, 13, 10, 0)
    mset(jac, 13, 11, 0)
    mset(jac, 13, 12, vget(X, 0)*x5_13 + vget(X, 2)*x21_11)
    mset(jac, 13, 13, -vget(X, 0)*x13_13*x15_11 - vget(X, 1)*x20_11)
    mset(jac, 13, 14, (-1.0409203801861816e-115*exp((22.915965629999999)*log(abs(T)))*vget(X, 0)*x14_13 + vget(X, 0)*vget(X, 12)*((((
       x22_11
    )) if truthy((x2_13)) else ((
       -1875129.1683687614*exp((-2.5)*log(abs(T)))*x4_13 + x22_11 + 587469852277.90271*x23_11*x4_13 + 54.282214350476039*x23_11*exp(-563932.20901156683*x3_13)
    )))) - vget(X, 0)*x16_11*(-0.030054336600000002*x10_13*x3_13 + 0.001447065312*x11_13*x3_13 - 2.9193291280000001e-5*x12_13*x3_13 - 21.506460400000002*x3_13*x6_13 + 9.1741162500000009*x3_13*x7_13 - 2.2740475600000001*x3_13*x8_13 + 0.33976956150000004*x3_13*x9_13) - vget(X, 1)*vget(X, 13)*((((
       0.00016065*exp((-2.75)*log(abs(T)))*x18_11 - 9.4499999999999994e-10*exp((-1.75)*log(abs(T)))*x18_11
    )) if truthy((x19_11)) else ((
       1.8960000000000001e-36*exp((3.7400000000000002)*log(abs(T)))
    )))) + 7.2084342424042629e-17*vget(X, 12)*vget(X, 2)*x17_11)/(vget(X, 0)*x25_9 + vget(X, 1)*x25_9 + vget(X, 10)*x26_9 + vget(X, 11)*x25_9 + vget(X, 12)*x25_9 + vget(X, 13)*x25_9 + vget(X, 2)*x25_9 + vget(X, 3)*x25_9 + vget(X, 4)*x25_9 + vget(X, 5)*x25_9 + vget(X, 6)*x26_9 + vget(X, 7)*x25_9 + vget(X, 8)*x26_9 + vget(X, 9)*x26_9))
    var x0_14 = 0.00013612213614898791*vget(X, 0) + 0.24994102282436673*vget(X, 1) + 0.75007714496081457*vget(X, 10) + 0.99972775572710437*vget(X, 11) + 0.99986387786355213*vget(X, 12) + vget(X, 13) + 0.25007714496081457*vget(X, 2) + 0.25021326709726244*vget(X, 3) + 0.49986387786355219*vget(X, 4) + 0.5*vget(X, 5) + 0.50001816778518127*vget(X, 6) + 0.50013612213644787*vget(X, 7) + 0.50015428992162914*vget(X, 8) + 0.7499410228243667*vget(X, 9)
    var x1_14 = ((x0_14)*(x0_14))
    var x2_14 = 1.0/x1_14
    var x3_14 = vget(X, 1) + vget(X, 12) + vget(X, 4)
    var x4_14 = 4.0*vget(X, 11) + x3_14
    var x5_14 = sqrt(T)
    var x6_14 = 2.1299999999999999e-27*x5_14
    var x7_14 = x4_14*x6_14
    var x8_14 = 1.0/T
    var x9_14 = exp(-102000.0*x8_14)
    var x10_14 = vget(X, 8)*x9_14
    var x11_14 = 3.1438547368704001e-21*exp((0.34999999999999998)*log(abs(T)))
    var x12_14 = x10_14*x11_14
    var x13_14 = 2.73*z + 2.73
    var x14_14 = 5.6500000000000001e-36*((((z + 1.0)*(z + 1.0)))*(((z + 1.0)*(z + 1.0))))
    var x15_12 = x14_14*(T - x13_14)
    var x16_12 = T <= 10
    var x17_12 = 1.5499999999999999e-26*((((
       2.3157944032250755
    )) if truthy((x16_12)) else ((
       exp((0.36470000000000002)*log(abs(T)))
    ))))
    var x18_12 = vget(X, 12)*x17_12
    var x19_12 = sqrt(T)
    var x20_12 = ((((
       sqrt(10.0)
    )) if truthy((x16_12)) else ((
       x19_12
    ))))
    var x21_12 = 0.0031622776601683794*x20_12 + 1.0
    var x22_12 = 1.0/x21_12
    var x23_12 = vget(X, 2)*x22_12
    var x24_11 = ((((
       1.0/10.0
    )) if truthy((x16_12)) else ((
       x8_14
    ))))
    var x25_10 = exp(-118348.0*x24_11)
    var x26_10 = 7.4999999999999996e-19*x25_10
    var x27_9 = x23_12*x26_10
    var x28_9 = 6.3095734448019361e-5*((((
       5.011872336272722
    )) if truthy((x16_12)) else ((
       exp((0.69999999999999996)*log(abs(T)))
    )))) + 1.0
    var x29_8 = 1.0/x28_9
    var x30_7 = ((((
       0.63095734448019325
    )) if truthy((x16_12)) else ((
       exp((-0.20000000000000001)*log(abs(T)))
    ))))
    var x31_7 = x29_8*x30_7
    var x32_7 = x20_12*x31_7
    var x33_7 = 3.4635323838154264e-26*x32_7
    var x34_7 = vget(X, 1)*x33_7
    var x35_7 = 1.3854129535261706e-25*x32_7
    var x36_7 = vget(X, 11)*x35_7
    var x37_7 = vget(X, 0)*vget(X, 12)
    var x38_7 = exp((-1.5)*log(abs(T)))
    var x39_7 = ((((
       0.031622776601683791
    )) if truthy((x16_12)) else ((
       x38_7
    ))))
    var x40_7 = 1.0 + 0.29999999999999999*exp(-94000.0*x24_11)
    var x41_7 = exp(-470000.0*x24_11)
    var x42_7 = 1.24e-13*x40_7*x41_7
    var x43_6 = x39_7*x42_7
    var x44_5 = vget(X, 10) + vget(X, 2) + vget(X, 3) + vget(X, 9)
    var x45_4 = vget(X, 1) + 2.0*vget(X, 6) + 2.0*vget(X, 8) + x44_5
    var x46_4 = 1.0/x45_4
    var x47_4 = 1.0/x5_14
    var x48_4 = exp((-2)*log(abs(T)))
    var x49_4 = exp(-160000.0*x48_4)
    var x50_4 = vget(X, 2)*x49_4
    var x51_4 = exp(-12000.0/(T + 1200.0))
    var x52_4 = vget(X, 8)*x51_4
    var x53_4 = 1.0/(1.6000000000000001*x50_4 + 1.3999999999999999*x52_4)
    var x54_4 = x47_4*x53_4
    var x55_4 = x46_4*x54_4
    var x56_4 = 1.0/(1000000.0*x55_4 + 1.0)
    var x57_4 = vget(X, 0)*x22_12
    var x58_4 = exp(-473638.0*x24_11)
    var x59_4 = x58_4*((((
       0.4008667176273028
    )) if truthy((x16_12)) else ((
       exp((-0.39700000000000002)*log(abs(T)))
    ))))
    var x60_4 = 5.5399999999999998e-17*x59_4
    var x61_4 = x57_4*x60_4
    var x62_4 = ((((
       0.67810976749343443
    )) if truthy((x16_12)) else ((
       exp((-0.16869999999999999)*log(abs(T)))
    ))))
    var x63_4 = exp(-55338.0*x24_11)
    var x64_4 = x62_4*x63_4
    var x65_4 = 5.0099999999999997e-27*x64_4
    var x66_4 = ((vget(X, 0))*(vget(X, 0)))
    var x67_4 = vget(X, 12)*x22_12
    var x68_4 = x66_4*x67_4
    var x69_3 = exp(-13179.0*x24_11)
    var x70_3 = x62_4*x69_3
    var x71_3 = 9.1000000000000001e-27*x70_3
    var x72_3 = exp(-631515.0*x24_11)
    var x73_3 = 4.9500000000000001e-22*x72_3
    var x74_3 = vget(X, 12)*x73_3
    var x75_3 = x20_12*x57_4
    var x76_3 = exp(-285335.40000000002*x24_11)
    var x77_3 = 9.3799999999999993e-22*x76_3
    var x78_3 = vget(X, 13)*x77_3
    var x79_3 = exp(-157809.10000000001*x24_11)
    var x80_3 = x20_12*x79_3
    var x81_3 = 1.2700000000000001e-21*x80_3
    var x82_3 = x23_12*x81_3
    var x83_3 = ((vget(X, 2))*(vget(X, 2))*(vget(X, 2)))
    var x84_3 = exp((-0.25)*log(abs(T)))
    var x85_3 = 2.0000000000000002e-31*x47_4 + 6.0000000000000001e-32*x84_3
    var x86_3 = ((vget(X, 2))*(vget(X, 2)))
    var x87_3 = 2.5000000000000002e-32*x47_4 + 7.5000000000000001e-33*x84_3
    var x88_3 = 1.3500000000000001e-9*exp((0.098492999999999997)*log(abs(T))) + 4.4350199999999998e-10*exp((0.55610000000000004)*log(abs(T))) + 3.7408500000000004e-16*exp((2.1825999999999999)*log(abs(T)))
    var x89_2 = 0.0061910000000000003*exp((1.0461)*log(abs(T))) + 8.9711999999999997e-11*exp((3.0424000000000002)*log(abs(T))) + 3.2575999999999999e-14*exp((3.7740999999999998)*log(abs(T))) + 1.0
    var x90_2 = 1.0/x89_2
    var x91_2 = sqrt(Pi)
    var x92_2 = 1.0/x91_2
    var x93_2 = 1.3806479999999999e-16*vget(X, 0) + 1.3806479999999999e-16*vget(X, 1) + 1.3806479999999999e-16*vget(X, 10) + 1.3806479999999999e-16*vget(X, 11) + 1.3806479999999999e-16*vget(X, 12) + 1.3806479999999999e-16*vget(X, 13) + 1.3806479999999999e-16*vget(X, 2) + 1.3806479999999999e-16*vget(X, 3) + 1.3806479999999999e-16*vget(X, 4) + 1.3806479999999999e-16*vget(X, 5) + 1.3806479999999999e-16*vget(X, 6) + 1.3806479999999999e-16*vget(X, 7) + 1.3806479999999999e-16*vget(X, 8) + 1.3806479999999999e-16*vget(X, 9)
    var x94_2 = 9.1093818800000008e-28*vget(X, 0) + 1.6726215800000001e-24*vget(X, 1) + 5.01956503638e-24*vget(X, 10) + 6.6902431600000005e-24*vget(X, 11) + 6.6911540981899994e-24*vget(X, 12) + 6.6920650363799998e-24*vget(X, 13) + 1.6735325181900001e-24*vget(X, 2) + 1.6744434563800001e-24*vget(X, 3) + 3.3451215800000003e-24*vget(X, 4) + 3.3460325181899999e-24*vget(X, 5) + 3.3461540981899999e-24*vget(X, 6) + 3.3469434563800003e-24*vget(X, 7) + 3.3470650363800003e-24*vget(X, 8) + 5.0186540981899997e-24*vget(X, 9)
    var x95_2 = 1.0/x94_2
    var x96_2 = exp((-1.0/2.0)*log(abs(x95_2)))
    var x97_2 = ((vget(X, 8))*(vget(X, 8)))
    var x98_2 = 1.1800000000000001e-10*exp(-69500.0*x8_14)
    var x99_2 = log(0.0001*T)
    var x100_2 = Log10
    var x101_2 = 1.0/x100_2
    var x102_2 = exp((-2)*log(abs(x100_2)))
    var x103_2 = exp((1.3*x101_2*x99_2 - 1.6200000000000001*x102_2*((x99_2)*(x99_2)) - 4.8449999999999998)*log(abs(10.0)))
    var x104_2 = x103_2*x45_4
    var x105_2 = x104_2 + 1.0
    var x106_2 = 1.0/x105_2
    var x107_2 = 1.0*x106_2
    var x108_2 = exp((x107_2)*log(abs(x98_2)))
    var x109_2 = 1.0 - exp(-6000.0*x8_14)
    var x110_2 = 52000.0*x8_14
    var x111_2 = exp(-x110_2)
    var x112_2 = x109_2*x111_2
    var x113_2 = 8.1250000000000003e-8*x112_2*x47_4
    var x114_2 = 1.0 - x107_2
    var x115_2 = exp((x114_2)*log(abs(x113_2)))
    var x116_2 = x108_2*x115_2
    var x117_2 = x116_2*x97_2
    var x118_2 = 7.1777505408000004e-12*x117_2
    var x119_2 = log(T)
    var x120_2 = ((x119_2)*(x119_2))
    var x121_2 = x102_2*x120_2
    var x122_2 = -4.8909149999999997*x101_2*x119_2 + 0.47490300000000002*x121_2 - 133.82830000000001*x8_14
    var x123_2 = exp((x122_2 + 14.82123)*log(abs(10.0)))
    var x124_2 = 1.0/x123_2
    var x125_2 = x124_2*x45_4
    var x126_2 = exp(-0.0022727272727272726*T)
    var x127_2 = exp(-0.00054054054054054055*T)
    var x128_2 = -2.0563129999999998*x126_2 + 0.58640729999999996*x127_2 + 0.82274429999999998
    var x129_2 = exp((x128_2)*log(abs(x125_2)))
    var x130_2 = x129_2 + 1.0
    var x131_2 = 1.0/x130_2
    var x132_2 = x101_2*x119_2
    var x133_2 = 16780.950000000001*x8_14 + 1.0
    var x134_2 = powi_m3(x100_2)
    var x135_2 = ((x119_2)*(x119_2)*(x119_2))*x134_2
    var x136_2 = 40870.379999999997*x8_14 + 1.0
    var x137_2 = -69.700860000000006*x101_2*log(x136_2) + 4.6331670000000003*x135_2
    var x138_2 = 19.734269999999999*x101_2*log(x133_2) + 37.886913*x102_2*x120_2 - 14.509090000000008*x132_2 - x137_2 - 307.31920000000002
    var x139_2 = exp((x122_2 + 13.656822)*log(abs(10.0)))
    var x140_2 = 1.0/x139_2
    var x141_2 = x140_2*x45_4
    var x142_2 = exp((x128_2)*log(abs(x141_2)))
    var x143_2 = x142_2 + 1.0
    var x144_2 = 1.0/x143_2
    var x145_2 = exp((43.20243*x102_2*x120_2 - x131_2*x138_2 - 68.422430000000006*x132_2 - x137_2 - 2080.4099999999999*x144_2*x8_14 - 23705.700000000001*x8_14 - 178.4239)*log(abs(10.0)))
    var x146_2 = -2.4640089999999999*x101_2*x119_2 + 0.19859550000000001*x121_2 + 743.05999999999995*x8_14
    var x147_2 = exp((x146_2 + 9.3055640000000004)*log(abs(10.0)))
    var x148_2 = 1.0/x147_2
    var x149_2 = x148_2*x45_4
    var x150_2 = 2.9375070000000001*x126_2 + 0.23588480000000001*x127_2 + 0.75022860000000002
    var x151_2 = exp((x150_2)*log(abs(x149_2)))
    var x152_2 = x151_2 + 1.0
    var x153_2 = 1.0/x152_2
    var x154_2 = 14254.549999999999*x8_14 + 1.0
    var x155_2 = 27535.310000000001*x8_14 + 1.0
    var x156_2 = -21.360939999999999*x101_2*log(x155_2) + 0.25820969999999999*x135_2
    var x157_2 = 70.138370000000009*x101_2*x119_2 + 11.28215*x101_2*log(x154_2) - 4.7035149999999994*x121_2 - x156_2 - 203.11568
    var x158_2 = exp((x146_2 + 8.1313220000000008)*log(abs(10.0)))
    var x159_2 = 1.0/x158_2
    var x160_2 = x159_2*x45_4
    var x161_2 = exp((x150_2)*log(abs(x160_2)))
    var x162_2 = x161_2 + 1.0
    var x163_2 = 1.0/x162_2
    var x164_2 = exp((42.707410000000003*x101_2*x119_2 - 2.0273650000000001*x121_2 - x153_2*x157_2 - x156_2 - 1657.4099999999999*x163_2*x8_14 - 21467.790000000001*x8_14 - 142.7664)*log(abs(10.0)))
    var x165_1 = 7.1777505408000004e-12*x145_2 + 7.1777505408000004e-12*x164_2
    var x166_1 = vget(X, 8)*x165_1
    var x167_1 = T >= 10000.0
    var x168_1 = log(((((
       10000.0
    )) if truthy((x167_1)) else ((
       T
    )))))
    var x169_1 = exp((-4)*log(abs(x100_2)))
    var x170_1 = ((((x168_1)*(x168_1)))*(((x168_1)*(x168_1))))
    var x171_1 = ((x168_1)*(x168_1)*(x168_1))
    var x172_1 = ((x168_1)*(x168_1))
    var x173_1 = vget(X, 2) <= 0.01
    var x174_1 = ((((
       False
    )) if truthy((x173_1)) else ((
       vget(X, 2) >= 10000000000.0
    ))))
    var x175_1 = log(((((
       10000000000.0
    )) if truthy((x174_1)) else ((
       ((((
          0.01
       )) if truthy((x173_1)) else ((
          vget(X, 2)
       ))))
    )))))
    var x176_1 = ((((x175_1)*(x175_1)))*(((x175_1)*(x175_1))))
    var x177_1 = ((x175_1)*(x175_1)*(x175_1))
    var x178_1 = ((x175_1)*(x175_1))
    var x179_1 = x102_2*x168_1
    var x180_1 = powi_m5(x100_2)
    var x181_1 = x176_1*x180_1
    var x182_1 = x170_1*x180_1
    var x183_1 = x169_1*x177_1
    var x184_1 = x169_1*x171_1
    var x185_1 = x134_2*x178_1
    var x186_1 = x134_2*x172_1
    var x187_1 = exp((-8)*log(abs(x100_2)))
    var x188_1 = x170_1*x187_1
    var x189_1 = powi_m7(x100_2)
    var x190_1 = x171_1*x189_1
    var x191_1 = x170_1*x189_1
    var x192_1 = exp((-6)*log(abs(x100_2)))
    var x193_1 = x172_1*x192_1
    var x194_1 = x171_1*x192_1
    var x195_1 = x170_1*x192_1
    var x196_1 = x177_1*x180_1
    var x197_1 = x171_1*x180_1
    var x198_1 = x169_1*x178_1
    var x199_1 = exp((21.93385*x101_2*x168_1 + 0.92432999999999998*x101_2*x175_1 - 10.19097*x102_2*x172_1 + 0.54962*x102_2*x178_1 + 2.1990599999999998*x134_2*x171_1 - 0.076759999999999995*x134_2*x177_1 - 0.0036600000000000001*x168_1*x181_1 + 0.11864*x168_1*x183_1 - 1.06447*x168_1*x185_1 - 0.17333999999999999*x169_1*x170_1 + 0.0027499999999999998*x169_1*x176_1 - 0.073660000000000003*x172_1*x196_1 + 0.62343000000000004*x172_1*x198_1 + 0.77951999999999999*x175_1*x179_1 - 0.0083499999999999998*x175_1*x182_1 + 0.11711000000000001*x175_1*x184_1 - 0.54262999999999995*x175_1*x186_1 + 6.1920000000000003e-5*x176_1*x188_1 - 0.00066631000000000004*x176_1*x190_1 + 0.0025140000000000002*x176_1*x193_1 - 0.001482*x177_1*x191_1 + 0.017590000000000001*x177_1*x194_1 + 0.0106*x178_1*x195_1 - 0.13768*x178_1*x197_1 - 42.567880000000002)*log(abs(10.0)))
    var x200_1 = vget(X, 10)*x199_1
    var x201_1 = T >= x13_14
    var x202_1 = vget(X, 6) + vget(X, 8)
    var x203_1 = vget(X, 0) + vget(X, 11) + vget(X, 13) + vget(X, 5) + vget(X, 7) + x202_1 + x3_14 + x44_5
    var x204_1 = x203_1 <= 9.9999999999999993e-41
    var x205_1 = x94_2 >= 9.9999999999999998e-13
    var x206_1 = x94_2 >= 0.5
    var x207_1 = x94_2 <= 9.9999999999999993e-41
    var x208_1 = x19_12*x91_2
    var x209_1 = x208_1*x94_2
    var x210_1 = exp((2.1498900000000001 - 0.69317629274152892*x101_2)*log(abs(10.0)))*x209_1
    var x211_1 = 1.0000420000000001*x101_2
    var x212_1 = exp((x211_1*log(x94_2) + 2.1498900000000001)*log(abs(10.0)))*x209_1
    var x213_1 = 1.0/abs(x0_14)
    var x214_1 = sqrt(x203_1)
    var x215_1 = sqrt(x2_14*x203_1)
    var x216_1 = ((((
       4.8339620236294848e-32/((x210_1 + 2.1986273043946046e-56)*(x210_1 + 2.1986273043946046e-56)) >= 1.0
    )) if truthy((x204_1  and  x205_1  and  x206_1  and  x207_1)) else ((
       ((((
          4.8339620236294848e-32/((x212_1 + 2.1986273043946046e-56)*(x212_1 + 2.1986273043946046e-56)) >= 1.0
       )) if truthy((x204_1  and  x205_1  and  x207_1)) else ((
          ((((
             True
          )) if truthy((x204_1  and  x207_1)) else ((
             ((((
                216.48287161311649/((x210_1*x213_1 + 1.471335691176954e-39)*(x210_1*x213_1 + 1.471335691176954e-39)) >= 1.0
             )) if truthy((x204_1  and  x205_1  and  x206_1)) else ((
                ((((
                   216.48287161311649/((x212_1*x213_1 + 1.471335691176954e-39)*(x212_1*x213_1 + 1.471335691176954e-39)) >= 1.0
                )) if truthy((x204_1  and  x205_1)) else ((
                   ((((
                      True
                   )) if truthy((x204_1)) else ((
                      ((((
                         4.833962023629485e-72/((x210_1*x214_1 + 2.1986273043946045e-76)*(x210_1*x214_1 + 2.1986273043946045e-76)) >= 1.0
                      )) if truthy((x205_1  and  x206_1  and  x207_1)) else ((
                         ((((
                            4.833962023629485e-72/((x212_1*x214_1 + 2.1986273043946045e-76)*(x212_1*x214_1 + 2.1986273043946045e-76)) >= 1.0
                         )) if truthy((x205_1  and  x207_1)) else ((
                            ((((
                               True
                            )) if truthy((x207_1)) else ((
                               ((((
                                  2.1648287161311648e-38/((x210_1*x215_1 + 1.471335691176954e-59)*(x210_1*x215_1 + 1.471335691176954e-59)) >= 1.0
                               )) if truthy((x205_1  and  x206_1)) else ((
                                  ((((
                                     2.1648287161311648e-38/((x212_1*x215_1 + 1.471335691176954e-59)*(x212_1*x215_1 + 1.471335691176954e-59)) >= 1.0
                                  )) if truthy((x205_1)) else ((
                                     True
                                  ))))
                               ))))
                            ))))
                         ))))
                      ))))
                   ))))
                ))))
             ))))
          ))))
       ))))
    ))))
    var x217_1 = exp((x211_1*log(((((
       0.5
    )) if truthy((x206_1)) else ((
       x94_2
    ))))) + 2.1498900000000001)*log(abs(10.0)))
    var x218_1 = ((((
       x217_1
    )) if truthy((x205_1)) else ((
       0.0
    ))))
    var x219_1 = ((((
       9.9999999999999993e-41
    )) if truthy((x204_1)) else ((
       x203_1
    ))))
    var x220_1 = ((((
       1.0e+80
    )) if truthy((x207_1)) else ((
       2.232953576238777e+46*x2_14
    ))))
    var x221_1 = sqrt(x219_1*x220_1)
    var x222_1 = x218_1*x221_1
    var x223_1 = x209_1*x222_1
    var x224_1 = x223_1 + 2.1986273043946046e-36
    var x225_1 = ((((
       1.0
    )) if truthy((x216_1)) else ((
       483396202.36294854/((x224_1)*(x224_1))
    ))))
    var x226_1 = ((((T)*(T)))*(((T)*(T))))
    var x227_1 = x218_1*x226_1
    var x228_1 = x225_1*x227_1
    var x229_1 = 0.00022681492*x94_2
    var x230_1 = T < 2.0
    var x231_1 = 1.2500000000000001e-10*vget(X, 0) + 1.2500000000000001e-10*vget(X, 1) + 1.2500000000000001e-10*vget(X, 10) + 1.2500000000000001e-10*vget(X, 11) + 1.2500000000000001e-10*vget(X, 12) + 1.2500000000000001e-10*vget(X, 13) + 1.2500000000000001e-10*vget(X, 2) + 1.2500000000000001e-10*vget(X, 3) + 1.2500000000000001e-10*vget(X, 4) + 1.2500000000000001e-10*vget(X, 5) + 1.2500000000000001e-10*vget(X, 6) + 1.2500000000000001e-10*vget(X, 7) + 1.2500000000000001e-10*vget(X, 8) + 1.2500000000000001e-10*vget(X, 9) <= 9.9999999999999993e-41
    var x232_1 = 28601.610899577994*exp((-0.45000000000000001)*log(abs(x203_1)))
    var x233_1 = ((((
       True
    )) if truthy((x231_1)) else ((
       x232_1 >= 1.0
    ))))
    var x234_1 = ((((
       1.0
    )) if truthy((x233_1)) else ((
       ((((
          1.000000000000001e+18
       )) if truthy((x231_1)) else ((
          x232_1
       ))))
    ))))
    var x235_1 = exp((25.0*x101_2)*log(abs(T)))
    var x236_1 = 1.0/x235_1
    var x237_1 = 1.0/(2.3538526683701997e+17*x236_1 + 10.0)
    var x238_1 = 1.0/(1.6889118802245084e-48*x235_1 + 10.0)
    var x239_1 = exp((20000.0*x237_1*x238_1 - 200.0)*log(abs(10.0)))
    var x240_0 = log(0.001*T)
    var x241_0 = ((x240_0)*(x240_0)*(x240_0)*(x240_0)*(x240_0))
    var x242_0 = x180_1*x241_0
    var x243_0 = ((((x240_0)*(x240_0)))*(((x240_0)*(x240_0))))
    var x244_0 = x169_1*x243_0
    var x245_0 = ((x240_0)*(x240_0)*(x240_0))
    var x246_0 = ((x240_0)*(x240_0))
    var x247_0 = x102_2*x246_0
    var x248_0 = exp((2.0943374000000001*x101_2*x240_0 + 0.43693353000000001*x134_2*x245_0 - 0.033638326000000003*x242_0 - 0.14913216000000001*x244_0 - 0.77151435999999995*x247_0 - 23.962112000000001)*log(abs(10.0)))
    var x249_0 = x239_1*x248_0
    var x250_0 = vget(X, 8)*x249_0
    var x251_0 = x101_2*x240_0
    var x252_0 = x134_2*x245_0
    var x253_0 = exp((0.19191374999999999*x242_0 - 0.16596184*x244_0 - 0.81520437999999995*x247_0 + 2.1892372*x251_0 + 0.29036281000000003*x252_0 - 23.689236999999999)*log(abs(10.0)))
    var x254_0 = vget(X, 13)*x253_0
    var x255_0 = T <= 10000.0
    var x256_0 = T > 10.0
    var x257_0 = x255_0  and  x256_0
    var x258_0 = exp((16.666666666666664*x101_2)*log(abs(T)))
    var x259_0 = 1.0/x258_0
    var x260_0 = 1.0/(785.77199422741614*x259_0 + 10.0)
    var x261_0 = 1.0/(5.0592917094448065e-34*x258_0 + 10.0)
    var x262_0 = exp((20000.0*x260_0*x261_0 - 200.0)*log(abs(10.0)))
    var x263_0 = 1.002560385050777e-22*x262_0
    var x264_0 = vget(X, 13)*x263_0
    var x265_0 = exp((0.32168730000000001*x242_0 - 0.51002221000000003*x244_0 + 0.015391166*x247_0 + 1.5714710999999999*x251_0 - 0.23619984999999999*x252_0 - 22.089523)*log(abs(10.0)))
    var x266_0 = vget(X, 1)*x265_0
    var x267_0 = 1.1825091393820599e-21*x262_0
    var x268_0 = vget(X, 1)*x267_0
    var x269_0 = exp((3.8479610000000002*x242_0 + 20.159831000000001*x244_0 + 58.145166000000003*x247_0 + 37.383713*x251_0 + 48.656103000000002*x252_0 - 16.818342000000001)*log(abs(10.0)))
    var x270_0 = vget(X, 2)*x269_0
    var x271_0 = T <= 100.0
    var x272_0 = exp((3.5692468000000002*x101_2*x240_0 - 4.2519023000000002*x242_0 - 21.328264000000001*x244_0 - 11.33286*x247_0 - 27.850082*x252_0 - 24.311209000000002)*log(abs(10.0)))
    var x273_0 = vget(X, 2)*x272_0
    var x274_0 = T <= 1000.0
    var x275_0 = exp((1.5538288*x242_0 - 5.5108049000000001*x244_0 - 3.7209846*x247_0 + 4.6450521*x251_0 + 5.9369081000000001*x252_0 - 24.311209000000002)*log(abs(10.0)))
    var x276_0 = vget(X, 2)*x275_0
    var x277_0 = T <= 6000.0
    var x278_0 = exp((17.997580222853362*x101_2)*log(abs(T)))
    var x279_0 = 1.0/x278_0
    var x280_0 = 1.0/(2973.7534532281375*x279_0 + 10.0)
    var x281_0 = 1.0/(1.3368457736780898e-34*x278_0 + 10.0)
    var x282_0 = 1.8623144679125181e-22*exp((20000.0*x280_0*x281_0 - 200.0)*log(abs(10.0)))
    var x283_0 = vget(X, 2)*x282_0
    var x284_0 = exp((8)*log(abs(x240_0)))
    var x285_0 = x187_1*x284_0
    var x286_0 = ((x240_0)*(x240_0)*(x240_0)*(x240_0)*(x240_0)*(x240_0)*(x240_0))
    var x287_0 = x189_1*x286_0
    var x288_0 = exp((6)*log(abs(x240_0)))
    var x289_0 = x192_1*x288_0
    var x290_0 = exp((983.67575999999997*x242_0 + 734.71650999999997*x244_0 + 96.743155000000002*x247_0 + 16.815729999999999*x251_0 + 343.1918*x252_0 + 70.609154000000004*x285_0 + 364.14445999999998*x287_0 + 801.81246999999996*x289_0 - 21.928795999999998)*log(abs(10.0)))
    var x291_0 = vget(X, 0)*x290_0
    var x292_0 = T <= 500.0
    var x293_0 = T > 100
    var x294_0 = x292_0  and  x293_0
    var x295_0 = exp((-8.8077017000000009*x242_0 - 4.7274035999999997*x244_0 + 0.93310621999999999*x247_0 + 1.6802758*x251_0 + 4.0406627000000004*x252_0 - 6.3701156000000001*x285_0 + 6.4380698000000001*x287_0 + 8.9167182999999994*x289_0 - 22.921188999999998)*log(abs(10.0)))
    var x296_0 = vget(X, 0)*x295_0
    var x297_0 = T > 500.0
    var x298_0 = x239_1*((((
       x291_0
    )) if truthy((x294_0)) else (((((
       x296_0
    )) if truthy((x297_0)) else ((
       0
    )))))))
    var x299_0 = x250_0 + x298_0 + ((((
       x254_0
    )) if truthy((x257_0)) else ((
       x264_0
    )))) + ((((
       x266_0
    )) if truthy((x257_0)) else ((
       x268_0
    )))) + ((((
       x270_0
    )) if truthy((x271_0)) else (((((
       x273_0
    )) if truthy((x274_0)) else (((((
       x276_0
    )) if truthy((x277_0)) else ((
       x283_0
    ))))))))))
    var x300_0 = exp(-11700.0*x8_14)
    var x301_0 = exp(-5860.0*x8_14)
    var x302_0 = exp(-510.0*x8_14)
    var x303_0 = 6.0142468035272636e-8*exp((2.1000000000000001)*log(abs(T))) + 1.0
    var x304_0 = ((T)*(T)*(T))
    var x305_0 = 1.0/x304_0
    var x306_0 = exp(-2197000.0*x305_0)
    var x307_0 = x306_0/x303_0
    var x308_0 = 4.985670872372847e-33*exp((3.7599999999999998)*log(abs(T)))*x307_0 + 1.6e-18*x300_0 + 6.7e-19*x301_0 + 3.0e-24*x302_0
    var x309_0 = T < 2000.0
    var x310_0 = exp((5.0194035000000001*x101_2*x240_0 + 2.4714160999999999*x169_1*x243_0 + 5.4710749999999999*x180_1*x241_0 + 1.8161874*x187_1*x284_0 - 1.5738805*x247_0 - 4.7155769000000003*x252_0 - 2.2148338000000001*x287_0 - 3.9467355999999998*x289_0 - 20.584225)*log(abs(10.0)))
    var x311_0 = 0.00020000000000000001*T
    var x312_0 = x311_0 - 6.0
    var x313_0 = x312_0 >= 300.0
    var x314_0 = exp(((((
       300.0
    )) if truthy((x313_0)) else ((
       x312_0
    )))))
    var x315_0 = x314_0 + 1.0
    var x316_0 = ((((
       x308_0
    )) if truthy((x309_0)) else (((((
       x310_0
    )) if truthy((x255_0)) else ((
       5.5313336794064847e-19/x315_0
    )))))))
    var x317_0 = x299_0 + x316_0
    var x318_0 = 1.0/x317_0
    var x319_0 = x316_0*x318_0
    var x320_0 = x308_0 >= 1.0e-99
    var x321_0 = x239_1*x291_0
    var x322_0 = x250_0 + x270_0
    var x323_0 = x254_0 + x266_0
    var x324_0 = x322_0 + x323_0
    var x325_0 = x321_0 + x324_0 >= 1.0e-99
    var x326_0 = x250_0 + x323_0
    var x327_0 = x273_0 + x326_0
    var x328_0 = x321_0 + x327_0 >= 1.0e-99
    var x329_0 = x276_0 + x326_0
    var x330_0 = x321_0 + x329_0 >= 1.0e-99
    var x331_0 = x283_0 + x326_0
    var x332_0 = x321_0 + x331_0 >= 1.0e-99
    var x333_0 = x264_0 + x268_0
    var x334_0 = x322_0 + x333_0
    var x335_0 = x321_0 + x334_0 >= 1.0e-99
    var x336_0 = x250_0 + x333_0
    var x337_0 = x273_0 + x336_0
    var x338_0 = x321_0 + x337_0 >= 1.0e-99
    var x339_0 = x276_0 + x336_0
    var x340_0 = x321_0 + x339_0 >= 1.0e-99
    var x341_0 = x283_0 + x336_0
    var x342_0 = x321_0 + x341_0 >= 1.0e-99
    var x343_0 = x239_1*x296_0
    var x344_0 = x324_0 + x343_0 >= 1.0e-99
    var x345_0 = x327_0 + x343_0 >= 1.0e-99
    var x346_0 = x329_0 + x343_0 >= 1.0e-99
    var x347_0 = x331_0 + x343_0 >= 1.0e-99
    var x348_0 = x334_0 + x343_0 >= 1.0e-99
    var x349_0 = x337_0 + x343_0 >= 1.0e-99
    var x350_0 = x339_0 + x343_0 >= 1.0e-99
    var x351_0 = x341_0 + x343_0 >= 1.0e-99
    var x352_0 = x324_0 >= 1.0e-99
    var x353_0 = x327_0 >= 1.0e-99
    var x354_0 = x329_0 >= 1.0e-99
    var x355_0 = x331_0 >= 1.0e-99
    var x356_0 = x334_0 >= 1.0e-99
    var x357_0 = x337_0 >= 1.0e-99
    var x358_0 = x339_0 >= 1.0e-99
    var x359_0 = x341_0 >= 1.0e-99
    var x360_0 = x310_0 >= 1.0e-99
    var x361_0 = 5.5313336794064847e-19/(0.0024787521766663585*exp(x311_0) + 1.0) >= 1.0e-99
    var x362_0 = ((((
       x320_0  and  x325_0
    )) if truthy((x255_0  and  x256_0  and  x271_0  and  x292_0  and  x293_0  and  x309_0)) else ((
       ((((
          x320_0  and  x328_0
       )) if truthy((x255_0  and  x256_0  and  x274_0  and  x292_0  and  x293_0  and  x309_0)) else ((
          ((((
             x320_0  and  x330_0
          )) if truthy((x255_0  and  x256_0  and  x277_0  and  x292_0  and  x293_0  and  x309_0)) else ((
             ((((
                x320_0  and  x332_0
             )) if truthy((x255_0  and  x256_0  and  x292_0  and  x293_0  and  x309_0)) else ((
                ((((
                   x320_0  and  x335_0
                )) if truthy((x271_0  and  x292_0  and  x293_0  and  x309_0)) else ((
                   ((((
                      x320_0  and  x338_0
                   )) if truthy((x274_0  and  x292_0  and  x293_0  and  x309_0)) else ((
                      ((((
                         x320_0  and  x340_0
                      )) if truthy((x277_0  and  x292_0  and  x293_0  and  x309_0)) else ((
                         ((((
                            x320_0  and  x342_0
                         )) if truthy((x292_0  and  x293_0  and  x309_0)) else ((
                            ((((
                               x320_0  and  x344_0
                            )) if truthy((x255_0  and  x256_0  and  x271_0  and  x297_0  and  x309_0)) else ((
                               ((((
                                  x320_0  and  x345_0
                               )) if truthy((x255_0  and  x256_0  and  x274_0  and  x297_0  and  x309_0)) else ((
                                  ((((
                                     x320_0  and  x346_0
                                  )) if truthy((x255_0  and  x256_0  and  x277_0  and  x297_0  and  x309_0)) else ((
                                     ((((
                                        x320_0  and  x347_0
                                     )) if truthy((x255_0  and  x256_0  and  x297_0  and  x309_0)) else ((
                                        ((((
                                           x320_0  and  x348_0
                                        )) if truthy((x271_0  and  x297_0  and  x309_0)) else ((
                                           ((((
                                              x320_0  and  x349_0
                                           )) if truthy((x274_0  and  x297_0  and  x309_0)) else ((
                                              ((((
                                                 x320_0  and  x350_0
                                              )) if truthy((x277_0  and  x297_0  and  x309_0)) else ((
                                                 ((((
                                                    x320_0  and  x351_0
                                                 )) if truthy((x297_0  and  x309_0)) else ((
                                                    ((((
                                                       x320_0  and  x352_0
                                                    )) if truthy((x255_0  and  x256_0  and  x271_0  and  x309_0)) else ((
                                                       ((((
                                                          x320_0  and  x353_0
                                                       )) if truthy((x255_0  and  x256_0  and  x274_0  and  x309_0)) else ((
                                                          ((((
                                                             x320_0  and  x354_0
                                                          )) if truthy((x255_0  and  x256_0  and  x277_0  and  x309_0)) else ((
                                                             ((((
                                                                x320_0  and  x355_0
                                                             )) if truthy((x255_0  and  x256_0  and  x309_0)) else ((
                                                                ((((
                                                                   x320_0  and  x356_0
                                                                )) if truthy((x271_0  and  x309_0)) else ((
                                                                   ((((
                                                                      x320_0  and  x357_0
                                                                   )) if truthy((x274_0  and  x309_0)) else ((
                                                                      ((((
                                                                         x320_0  and  x358_0
                                                                      )) if truthy((x277_0  and  x309_0)) else ((
                                                                         ((((
                                                                            x320_0  and  x359_0
                                                                         )) if truthy((x309_0)) else ((
                                                                            ((((
                                                                               x325_0  and  x360_0
                                                                            )) if truthy((x255_0  and  x256_0  and  x271_0  and  x292_0  and  x293_0)) else ((
                                                                               ((((
                                                                                  x328_0  and  x360_0
                                                                               )) if truthy((x255_0  and  x256_0  and  x274_0  and  x292_0  and  x293_0)) else ((
                                                                                  ((((
                                                                                     x330_0  and  x360_0
                                                                                  )) if truthy((x255_0  and  x256_0  and  x277_0  and  x292_0  and  x293_0)) else ((
                                                                                     ((((
                                                                                        x332_0  and  x360_0
                                                                                     )) if truthy((x255_0  and  x256_0  and  x292_0  and  x293_0)) else ((
                                                                                        ((((
                                                                                           x335_0  and  x360_0
                                                                                        )) if truthy((x255_0  and  x271_0  and  x292_0  and  x293_0)) else ((
                                                                                           ((((
                                                                                              x338_0  and  x360_0
                                                                                           )) if truthy((x255_0  and  x274_0  and  x292_0  and  x293_0)) else ((
                                                                                              ((((
                                                                                                 x340_0  and  x360_0
                                                                                              )) if truthy((x255_0  and  x277_0  and  x292_0  and  x293_0)) else ((
                                                                                                 ((((
                                                                                                    x342_0  and  x360_0
                                                                                                 )) if truthy((x255_0  and  x292_0  and  x293_0)) else ((
                                                                                                    ((((
                                                                                                       x344_0  and  x360_0
                                                                                                    )) if truthy((x255_0  and  x256_0  and  x271_0  and  x297_0)) else ((
                                                                                                       ((((
                                                                                                          x345_0  and  x360_0
                                                                                                       )) if truthy((x255_0  and  x256_0  and  x274_0  and  x297_0)) else ((
                                                                                                          ((((
                                                                                                             x346_0  and  x360_0
                                                                                                          )) if truthy((x255_0  and  x256_0  and  x277_0  and  x297_0)) else ((
                                                                                                             ((((
                                                                                                                x347_0  and  x360_0
                                                                                                             )) if truthy((x255_0  and  x256_0  and  x297_0)) else ((
                                                                                                                ((((
                                                                                                                   x348_0  and  x360_0
                                                                                                                )) if truthy((x255_0  and  x271_0  and  x297_0)) else ((
                                                                                                                   ((((
                                                                                                                      x349_0  and  x360_0
                                                                                                                   )) if truthy((x255_0  and  x274_0  and  x297_0)) else ((
                                                                                                                      ((((
                                                                                                                         x350_0  and  x360_0
                                                                                                                      )) if truthy((x255_0  and  x277_0  and  x297_0)) else ((
                                                                                                                         ((((
                                                                                                                            x351_0  and  x360_0
                                                                                                                         )) if truthy((x255_0  and  x297_0)) else ((
                                                                                                                            ((((
                                                                                                                               x352_0  and  x360_0
                                                                                                                            )) if truthy((x255_0  and  x256_0  and  x271_0)) else ((
                                                                                                                               ((((
                                                                                                                                  x353_0  and  x360_0
                                                                                                                               )) if truthy((x255_0  and  x256_0  and  x274_0)) else ((
                                                                                                                                  ((((
                                                                                                                                     x354_0  and  x360_0
                                                                                                                                  )) if truthy((x255_0  and  x256_0  and  x277_0)) else ((
                                                                                                                                     ((((
                                                                                                                                        x355_0  and  x360_0
                                                                                                                                     )) if truthy((x257_0)) else ((
                                                                                                                                        ((((
                                                                                                                                           x356_0  and  x360_0
                                                                                                                                        )) if truthy((x255_0  and  x271_0)) else ((
                                                                                                                                           ((((
                                                                                                                                              x357_0  and  x360_0
                                                                                                                                           )) if truthy((x255_0  and  x274_0)) else ((
                                                                                                                                              ((((
                                                                                                                                                 x358_0  and  x360_0
                                                                                                                                              )) if truthy((x255_0  and  x277_0)) else ((
                                                                                                                                                 ((((
                                                                                                                                                    x359_0  and  x360_0
                                                                                                                                                 )) if truthy((x255_0)) else ((
                                                                                                                                                    ((((
                                                                                                                                                       False
                                                                                                                                                    )) if truthy((x313_0  and  (x271_0  or  x313_0)  and  (x274_0  or  x313_0)  and  (x277_0  or  x313_0)  and  (x292_0  or  x313_0)  and  (x293_0  or  x313_0)  and  (x297_0  or  x313_0)  and  (x271_0  or  x274_0  or  x313_0)  and  (x271_0  or  x277_0  or  x313_0)  and  (x271_0  or  x292_0  or  x313_0)  and  (x271_0  or  x293_0  or  x313_0)  and  (x271_0  or  x297_0  or  x313_0)  and  (x274_0  or  x277_0  or  x313_0)  and  (x274_0  or  x292_0  or  x313_0)  and  (x274_0  or  x293_0  or  x313_0)  and  (x274_0  or  x297_0  or  x313_0)  and  (x277_0  or  x292_0  or  x313_0)  and  (x277_0  or  x293_0  or  x313_0)  and  (x277_0  or  x297_0  or  x313_0)  and  (x292_0  or  x293_0  or  x313_0)  and  (x293_0  or  x297_0  or  x313_0)  and  (x271_0  or  x274_0  or  x277_0  or  x313_0)  and  (x271_0  or  x274_0  or  x292_0  or  x313_0)  and  (x271_0  or  x274_0  or  x293_0  or  x313_0)  and  (x271_0  or  x274_0  or  x297_0  or  x313_0)  and  (x271_0  or  x277_0  or  x292_0  or  x313_0)  and  (x271_0  or  x277_0  or  x293_0  or  x313_0)  and  (x271_0  or  x277_0  or  x297_0  or  x313_0)  and  (x271_0  or  x292_0  or  x293_0  or  x313_0)  and  (x271_0  or  x293_0  or  x297_0  or  x313_0)  and  (x274_0  or  x277_0  or  x292_0  or  x313_0)  and  (x274_0  or  x277_0  or  x293_0  or  x313_0)  and  (x274_0  or  x277_0  or  x297_0  or  x313_0)  and  (x274_0  or  x292_0  or  x293_0  or  x313_0)  and  (x274_0  or  x293_0  or  x297_0  or  x313_0)  and  (x277_0  or  x292_0  or  x293_0  or  x313_0)  and  (x277_0  or  x293_0  or  x297_0  or  x313_0)  and  (x271_0  or  x274_0  or  x277_0  or  x292_0  or  x313_0)  and  (x271_0  or  x274_0  or  x277_0  or  x293_0  or  x313_0)  and  (x271_0  or  x274_0  or  x277_0  or  x297_0  or  x313_0)  and  (x271_0  or  x274_0  or  x292_0  or  x293_0  or  x313_0)  and  (x271_0  or  x274_0  or  x293_0  or  x297_0  or  x313_0)  and  (x271_0  or  x277_0  or  x292_0  or  x293_0  or  x313_0)  and  (x271_0  or  x277_0  or  x293_0  or  x297_0  or  x313_0)  and  (x274_0  or  x277_0  or  x292_0  or  x293_0  or  x313_0)  and  (x274_0  or  x277_0  or  x293_0  or  x297_0  or  x313_0)  and  (x271_0  or  x274_0  or  x277_0  or  x292_0  or  x293_0  or  x313_0)  and  (x271_0  or  x274_0  or  x277_0  or  x293_0  or  x297_0  or  x313_0))) else ((
                                                                                                                                                       ((((
                                                                                                                                                          x335_0  and  x361_0
                                                                                                                                                       )) if truthy((x271_0  and  x292_0  and  x293_0)) else ((
                                                                                                                                                          ((((
                                                                                                                                                             x338_0  and  x361_0
                                                                                                                                                          )) if truthy((x274_0  and  x292_0  and  x293_0)) else ((
                                                                                                                                                             ((((
                                                                                                                                                                x340_0  and  x361_0
                                                                                                                                                             )) if truthy((x277_0  and  x292_0  and  x293_0)) else ((
                                                                                                                                                                ((((
                                                                                                                                                                   x342_0  and  x361_0
                                                                                                                                                                )) if truthy((x294_0)) else ((
                                                                                                                                                                   ((((
                                                                                                                                                                      x348_0  and  x361_0
                                                                                                                                                                   )) if truthy((x271_0  and  x297_0)) else ((
                                                                                                                                                                      ((((
                                                                                                                                                                         x349_0  and  x361_0
                                                                                                                                                                      )) if truthy((x274_0  and  x297_0)) else ((
                                                                                                                                                                         ((((
                                                                                                                                                                            x350_0  and  x361_0
                                                                                                                                                                         )) if truthy((x277_0  and  x297_0)) else ((
                                                                                                                                                                            x351_0  and  x361_0
                                                                                                                                                                         ))))
                                                                                                                                                                      ))))
                                                                                                                                                                   ))))
                                                                                                                                                                ))))
                                                                                                                                                             ))))
                                                                                                                                                          ))))
                                                                                                                                                       ))))
                                                                                                                                                    ))))
                                                                                                                                                 ))))
                                                                                                                                              ))))
                                                                                                                                           ))))
                                                                                                                                        ))))
                                                                                                                                     ))))
                                                                                                                                  ))))
                                                                                                                               ))))
                                                                                                                            ))))
                                                                                                                         ))))
                                                                                                                      ))))
                                                                                                                   ))))
                                                                                                                ))))
                                                                                                             ))))
                                                                                                          ))))
                                                                                                       ))))
                                                                                                    ))))
                                                                                                 ))))
                                                                                              ))))
                                                                                           ))))
                                                                                        ))))
                                                                                     ))))
                                                                                  ))))
                                                                               ))))
                                                                            ))))
                                                                         ))))
                                                                      ))))
                                                                   ))))
                                                                ))))
                                                             ))))
                                                          ))))
                                                       ))))
                                                    ))))
                                                 ))))
                                              ))))
                                           ))))
                                        ))))
                                     ))))
                                  ))))
                               ))))
                            ))))
                         ))))
                      ))))
                   ))))
                ))))
             ))))
          ))))
       ))))
    ))))
    var x363_0 = ((((
       x299_0*x319_0
    )) if truthy((x362_0)) else ((
       0
    ))))
    var x364_0 = x234_1*x363_0
    var x365_0 = x2_14*(0.00084373771595996178*T*x92_2*x93_2*x96_2 - vget(X, 0)*x12_14 - vget(X, 0)*x15_12 - vget(X, 0)*x18_12 - vget(X, 0)*x27_9 - vget(X, 0)*x34_7 - vget(X, 0)*x36_7 - vget(X, 0)*x7_14 - vget(X, 0)*x82_3 - vget(X, 12)*x61_4 + 5.6556829037999995e-12*vget(X, 2)*vget(X, 3)*x56_4*x88_3*x90_2 + 1.75918975308e-21*vget(X, 2)*vget(X, 6)*x56_4 - vget(X, 2)*x166_1 + 7.1777505408000004e-12*vget(X, 8)*x56_4*x86_3*x87_3 - x118_2 - x228_1*x229_1 - x37_7*x43_6 + 7.1777505408000004e-12*x56_4*x83_3*x85_3 - x65_4*x68_4 - x68_4*x71_3 - x74_3*x75_3 - x75_3*x78_3 - ((((
       0
    )) if truthy((x230_1)) else ((
       vget(X, 8)*x364_0
    )))) - ((((
       x200_1
    )) if truthy((x201_1)) else ((
       0
    )))))
    var x366_0 = vget(X, 12)*x57_4
    var x367_0 = -1.1649047900646892e-19*T*x92_2*x96_2
    var x368_0 = x20_12*x22_12
    var x369_0 = 2.3026818015679518*x101_2*x217_1*((((
       2.0
    )) if truthy((x206_1)) else ((
       x95_2
    ))))
    var x370_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          9.1093818800000008e-28
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x371_0 = x225_1*x226_1*x229_1
    var x372_0 = x208_1*x222_1
    var x373_0 = 2*x209_1*x221_1
    var x374_0 = (1.0/2.0)*x220_1*((((
       0.0
    )) if truthy((x204_1)) else ((
       1.0
    ))))
    var x375_0 = powi_m3(x0_14)
    var x376_0 = (1.0/2.0)*x219_1
    var x377_0 = 2*x223_1*((((
       9.9999999999999996e-81
    )) if truthy((x207_1)) else ((
       4.4783734451139649e-47*x1_14
    ))))*((((
       1.0e+40
    )) if truthy((x204_1)) else ((
       1.0/x203_1
    ))))
    var x378_0 = 483396202.36294854/((x224_1)*(x224_1)*(x224_1))
    var x379_0 = x227_1*x229_1
    var x380_0 = vget(X, 8)*x363_0*((((
       0
    )) if truthy((x233_1)) else ((
       ((((
          0
       )) if truthy((x231_1)) else ((
          -12870.724904810098*exp((-1.45)*log(abs(x203_1)))
       ))))
    ))))
    var x381_0 = ((((
       x290_0
    )) if truthy((x294_0)) else (((((
       x295_0
    )) if truthy((x297_0)) else ((
       0
    )))))))
    var x382_0 = x299_0*x316_0/((x317_0)*(x317_0))
    var x383_0 = vget(X, 8)*x234_1
    var x384_0 = vget(X, 0)*x6_14
    var x385_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          1.6726215800000001e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x386_0 = ((((
       x265_0
    )) if truthy((x257_0)) else ((
       x267_0
    ))))
    var x387_0 = x142_2*x8_14/((x143_2)*(x143_2))
    var x388_0 = 4790.3210533157426*x387_0
    var x389_0 = x139_2*x140_2
    var x390_0 = x128_2*x46_4
    var x391_0 = x389_0*x390_0
    var x392_0 = x129_2*x138_2/((x130_2)*(x130_2))
    var x393_0 = 2.3025850929940459*x392_0
    var x394_0 = x123_2*x124_2
    var x395_0 = x390_0*x394_0
    var x396_0 = x161_2*x8_14/((x162_2)*(x162_2))
    var x397_0 = 3816.3275589792611*x396_0
    var x398_0 = x158_2*x159_2
    var x399_0 = x150_2*x46_4
    var x400_0 = x398_0*x399_0
    var x401_0 = x151_2*x157_2/((x152_2)*(x152_2))
    var x402_0 = 2.3025850929940459*x401_0
    var x403_0 = x147_2*x148_2
    var x404_0 = x399_0*x403_0
    var x405_0 = 7.1777505408000004e-12*vget(X, 8)
    var x406_0 = vget(X, 2)*x405_0
    var x407_0 = x406_0*(x145_2*(x388_0*x391_0 + x393_0*x395_0) + x164_2*(x397_0*x400_0 + x402_0*x404_0))
    var x408_0 = exp((-2)*log(abs(x105_2)))
    var x409_0 = x408_0*log(x113_2)
    var x410_0 = x103_2*x118_2
    var x411_0 = x409_0*x410_0
    var x412_0 = log(x98_2)
    var x413_0 = exp((-2)*log(abs(0.5*vget(X, 1) + 0.5*vget(X, 10) + 0.5*vget(X, 2) + 0.5*vget(X, 3) + 0.5*vget(X, 9) + x202_1)))
    var x414_0 = exp((-2)*log(abs(x55_4 + 9.9999999999999995e-7)))
    var x415_0 = -1.4139207259499998e-18*vget(X, 2)*vget(X, 3)*x413_0*x414_0*x47_4*x53_4*x88_3*x90_2 - 4.3979743826999997e-28*vget(X, 2)*vget(X, 6)*x413_0*x414_0*x47_4*x53_4 - 1.7944376352000002e-18*vget(X, 8)*x413_0*x414_0*x47_4*x53_4*x86_3*x87_3 - 7.1777505408000004e-12*x103_2*x108_2*x115_2*x408_0*x412_0*x97_2 + x407_0 + x411_0 - 1.7944376352000002e-18*x413_0*x414_0*x47_4*x53_4*x83_3*x85_3
    var x416_0 = x92_2*x96_2
    var x417_0 = T*x416_0
    var x418_0 = 1.1649047900646892e-19*x417_0
    var x419_0 = ((((
       x269_0
    )) if truthy((x271_0)) else (((((
       x272_0
    )) if truthy((x274_0)) else (((((
       x275_0
    )) if truthy((x277_0)) else ((
       x282_0
    ))))))))))
    var x420_0 = ((((
       0.0
    )) if truthy((x174_1)) else ((
       ((((
          0.0
       )) if truthy((x173_1)) else ((
          1.0
       ))))
    ))))*((((
       1.0e-10
    )) if truthy((x174_1)) else ((
       ((((
          100.0
       )) if truthy((x173_1)) else ((
          1.0/vget(X, 2)
       ))))
    ))))
    var x421_0 = x102_2*x175_1
    var x422_0 = x168_1*x420_0
    var x423_0 = x134_2*x175_1
    var x424_0 = x175_1*x420_0
    var x425_0 = x169_1*x172_1
    var x426_0 = x177_1*x420_0
    var x427_0 = x178_1*x420_0
    var x428_0 = x172_1*x180_1
    var x429_0 = 1.75918975308e-21*x56_4
    var x430_0 = x413_0*x54_4
    var x431_0 = x46_4*x47_4/((x50_4 + 0.87499999999999989*x52_4)*(x50_4 + 0.87499999999999989*x52_4))
    var x432_0 = x414_0*(250000.0*x430_0 + 624999.99999999988*x431_0*x49_4)
    var x433_0 = 7.1777505407999997e-24*x432_0
    var x434_0 = x83_3*x85_3
    var x435_0 = 1.4355501081600001e-11*vget(X, 8)
    var x436_0 = vget(X, 2)*vget(X, 6)
    var x437_0 = 1.7591897530800001e-33*x436_0
    var x438_0 = vget(X, 3)*x88_3*x90_2
    var x439_0 = 5.6556829037999995e-12*x56_4
    var x440_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          1.6735325181900001e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x441_0 = vget(X, 8)*x86_3*x87_3
    var x442_0 = vget(X, 2)*x438_0
    var x443_0 = 5.6556829037999991e-24*x442_0
    var x444_0 = x2_14*x417_0*x93_2*x94_2
    var x445_0 = x408_0*x412_0
    var x446_0 = -x407_0 + x410_0*x445_0 - x411_0
    var x447_0 = vget(X, 2)*x439_0*x90_2
    var x448_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          1.6744434563800001e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x449_0 = x414_0*x430_0
    var x450_0 = x434_0*x449_0
    var x451_0 = x436_0*x449_0
    var x452_0 = x441_0*x449_0
    var x453_0 = x442_0*x449_0
    var x454_0 = ((((
       0
    )) if truthy((x230_1)) else ((
       x380_0
    ))))
    var x455_0 = x418_0 - x454_0
    var x456_0 = x446_0 + 1.7944376352000002e-18*x450_0 + 4.3979743826999997e-28*x451_0 + 1.7944376352000002e-18*x452_0 + 1.4139207259499998e-18*x453_0 + x455_0
    var x457_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          3.3451215800000003e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x458_0 = x367_0 + x454_0
    var x459_0 = x384_0 + x458_0
    var x460_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          3.3460325181899999e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x461_0 = x406_0*(x145_2*(9580.6421066314851*x387_0*x391_0 + 4.6051701859880918*x392_0*x395_0) + x164_2*(7632.6551179585222*x396_0*x400_0 + 4.6051701859880918*x401_0*x404_0))
    var x462_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          3.3461540981899999e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x463_0 = 1.4355501081600001e-11*x103_2*x117_2
    var x464_0 = x409_0*x463_0
    var x465_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          3.3469434563800003e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x466_0 = 500000.0*x430_0 + 546874.99999999988*x431_0*x51_4
    var x467_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          3.3470650363800003e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x468_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          5.0186540981899997e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x469_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          5.01956503638e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x470_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          6.6902431600000005e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x471_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          6.6911540981899994e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x472_0 = x22_12*x66_4
    var x473_0 = ((((
       x369_0*((((
          0
       )) if truthy((x206_1)) else ((
          6.6920650363799998e-24
       ))))
    )) if truthy((x205_1)) else ((
       0
    ))))
    var x474_0 = ((((
       x253_0
    )) if truthy((x257_0)) else ((
       x263_0
    ))))
    var x475_0 = 2.0860422997526066e-16*x95_2
    var x476_0 = 3.4767371836380304e-16*x95_2
    var x477_0 = vget(X, 0)*x10_14
    var x478_0 = ((((
       0
    )) if truthy((x16_12)) else ((
       -x48_4
    ))))
    var x479_0 = x37_7*x39_7*x478_0
    var x480_0 = 3.4635323838154264e-26*vget(X, 1)
    var x481_0 = 1.0/x19_12
    var x482_0 = ((((
       0
    )) if truthy((x16_12)) else ((
       (1.0/2.0)*x481_0
    ))))
    var x483_0 = vget(X, 0)*x482_0
    var x484_0 = x31_7*x483_0
    var x485_0 = 1.3854129535261706e-25*vget(X, 11)
    var x486_0 = vget(X, 0)*x20_12
    var x487_0 = x29_8*x486_0*((((
       0
    )) if truthy((x16_12)) else ((
       -0.20000000000000001*exp((-1.2)*log(abs(T)))
    ))))
    var x488_0 = exp((-2.5)*log(abs(T)))
    var x489_0 = x482_0*x57_4
    var x490_0 = vget(X, 0)*x25_10
    var x491_0 = x482_0/((x21_12)*(x21_12))
    var x492_0 = vget(X, 2)*x491_0
    var x493_0 = x23_12*x478_0
    var x494_0 = x68_4*((((
       0
    )) if truthy((x16_12)) else ((
       -0.16869999999999999*exp((-1.1687000000000001)*log(abs(T)))
    ))))
    var x495_0 = x30_7*x486_0*((((
       0
    )) if truthy((x16_12)) else ((
       0.69999999999999996*exp((-0.30000000000000004)*log(abs(T)))
    ))))/((x28_9)*(x28_9))
    var x496_0 = exp((-1.25)*log(abs(T)))
    var x497_0 = x37_7*x491_0
    var x498_0 = vget(X, 12)*x491_0*x66_4
    var x499_0 = x478_0*x68_4
    var x500_0 = x478_0*x75_3
    var x501_0 = vget(X, 13)*x76_3
    var x502_0 = vget(X, 0)*x80_3
    var x503_0 = x414_0*(500000.0*x38_7*x46_4*x53_4 - 390624.99999999994*x431_0*(-512000.0*x305_0*x50_4 - 0.011666666666666665*x52_4/((0.00083333333333333339*T + 1)*(0.00083333333333333339*T + 1))))
    var x504_0 = x102_2*x8_14
    var x505_0 = 1.0*x104_2*(2.9933606208922598*x101_2*x8_14 - 7.460375701300709*x504_0*x99_2)
    var x506_0 = ((((
       0.0
    )) if truthy((x167_1)) else ((
       1.0
    ))))*((((
       0.0001
    )) if truthy((x167_1)) else ((
       x8_14
    ))))
    var x507_0 = x168_1*x506_0
    var x508_0 = x176_1*x506_0
    var x509_0 = x175_1*x506_0
    var x510_0 = x177_1*x506_0
    var x511_0 = x178_1*x506_0
    var x512_0 = x101_2*x8_14
    var x513_0 = x119_2*x504_0
    var x514_0 = x134_2*x8_14
    var x515_0 = x120_2*x514_0
    var x516_0 = x101_2*x48_4
    var x517_0 = x516_0/x136_2
    var x518_0 = 0.0046734386363636356*x126_2 - 0.00031697691891891889*x127_2
    var x519_0 = x128_2*(11.261747970100974*x101_2*x8_14 - 308.15104860073512*x48_4 - 2.1870091368363029*x513_0)
    var x520_0 = x516_0/x155_2
    var x521_0 = -0.0066761522727272725*x126_2 - 0.0001275052972972973*x127_2
    var x522_0 = x150_2*(1710.9588792001557*x48_4 + 5.6735903924031659*x512_0 - 0.91456607567139814*x513_0)
    var x523_0 = x240_0*x504_0
    var x524_0 = x187_1*x286_0*x8_14
    var x525_0 = x189_1*x288_0*x8_14
    var x526_0 = x192_1*x241_0*x8_14
    var x527_0 = x180_1*x243_0*x8_14
    var x528_0 = x169_1*x245_0*x8_14
    var x529_0 = x246_0*x514_0
    var x530_0 = ((((
       3.2860556719809434e-26*exp((-0.24000000000000021)*log(abs(T)))*x307_0 + 1.8746122480121903e-32*exp((2.7599999999999998)*log(abs(T)))*x307_0 - 6.2968615725975507e-40*exp((4.8599999999999994)*log(abs(T)))*x306_0/((x303_0)*(x303_0)) + 1.8719999999999998e-14*x300_0*x48_4 + 3.9261999999999998e-15*x301_0*x48_4 + 1.53e-21*x302_0*x48_4
    )) if truthy((x309_0)) else (((((
       x310_0*(11.55760367482214*x512_0 - 7.2479875549080308*x523_0 + 33.455408266588918*x524_0 - 35.698903039375494*x525_0 - 54.526167351293466*x526_0 + 62.988078688261993*x527_0 + 22.762583481781927*x528_0 - 32.574051224421225*x529_0)
    )) if truthy((x255_0)) else ((
       -5.5313336794064847e-19*x314_0*((((
          0
       )) if truthy((x313_0)) else ((
          0.00020000000000000001
       ))))/((x315_0)*(x315_0))
    )))))))
    var x531_0 = 4.8910985889961177e-12*x101_2*x236_1*x238_1*x8_14/((x236_1 + 4.2483542552915895e-17)*(x236_1 + 4.2483542552915895e-17)) - 1.9444316593927493e-44*x235_1*x237_1*x512_0/((1.6889118802245085e-49*x235_1 + 1)*(1.6889118802245085e-49*x235_1 + 1))
    var x532_0 = 976.7825399351309*x101_2*x259_0*x261_0*x8_14/((x259_0 + 0.012726338013398102)*(x259_0 + 0.012726338013398102)) - 3.8831498904253243e-30*x258_0*x260_0*x512_0/((5.0592917094448061e-35*x258_0 + 1)*(5.0592917094448061e-35*x258_0 + 1))
    var x533_0 = x239_1*((((
       x291_0*(38.719649225812766*x512_0 + 445.51869310442481*x523_0 + 1300.6686834345674*x524_0 + 5869.3152370465659*x525_0 + 11077.448644792414*x526_0 + 11324.985706577943*x527_0 + 6766.9891340104432*x528_0 + 2370.6849681533818*x529_0)
    )) if truthy((x294_0)) else (((((
       x296_0*(3.8689780091986448*x512_0 + 4.297112944704045*x523_0 - 117.34186576967058*x524_0 + 103.76942484394611*x525_0 + 123.18901581604325*x526_0 - 101.40241318979159*x527_0 - 43.540996231705549*x528_0 + 27.91190909651122*x529_0)
    )) if truthy((x297_0)) else ((
       0
    ))))))) + x250_0*x531_0 + x250_0*(4.8223900769399082*x101_2*x8_14 + 3.0182298984217999*x134_2*x246_0*x8_14 - 3.5529549287336835*x523_0 - 0.3872755400043702*x527_0 - 1.3735579540080118*x528_0) + x298_0*x531_0 + ((((
       x254_0*(5.0409049417480247*x512_0 - 3.7541549062629067*x523_0 + 2.2094886994529301*x527_0 - 1.528565035159452*x528_0 + 2.0057552335975877*x529_0)
    )) if truthy((x257_0)) else ((
       x264_0*x532_0
    )))) + ((((
       x266_0*(3.6184459289309552*x512_0 + 0.0708789387907936*x523_0 + 3.7035619079275177*x527_0 - 4.6974781513675152*x528_0 - 1.6316107607322889*x529_0)
    )) if truthy((x257_0)) else ((
       x268_0*x532_0
    )))) + ((((
       x270_0*(86.079180274567719*x512_0 + 267.76838492252847*x523_0 + 44.301288185112313*x527_0 + 185.67890535151699*x528_0 + 336.10445235294861*x529_0)
    )) if truthy((x271_0)) else (((((
       x273_0*(8.2184944748967013*x101_2*x8_14 - 52.189748993977005*x523_0 - 48.951834264235494*x527_0 - 196.44057098336626*x528_0 - 192.38155095558542*x529_0)
    )) if truthy((x274_0)) else (((((
       x276_0*(10.695627721640689*x512_0 - 17.135767342440825*x523_0 + 17.889115159724135*x527_0 - 50.756388852554174*x528_0 + 41.010708268606813*x529_0)
    )) if truthy((x277_0)) else ((
       x283_0*(278.71147075841589*x101_2*x279_0*x281_0*x8_14/((x279_0 + 0.003362753556164708)*(x279_0 + 0.003362753556164708)) - 1.1080034428212589e-30*x278_0*x280_0*x512_0/((1.3368457736780897e-35*x278_0 + 1)*(1.3368457736780897e-35*x278_0 + 1)))
    ))))))))))
    mset(jac, 14, 0, -2.0340826846270714e+19*x365_0 + x95_2*(8581161392004762.0*T*x2_14*x92_2*x93_2*x94_2*x96_2 - vget(X, 12)*x43_6 - x12_14 - x15_12 - x18_12 - 2.0661437223616499e-31*x228_1 - x27_9 - x34_7 - x36_7 - 1.0019999999999999e-26*x366_0*x64_4 - 1.82e-26*x366_0*x70_3 - x367_0 - x368_0*x74_3 - x368_0*x78_3 - x370_0*x371_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-x370_0*x373_0 - 1.8218763760000002e-27*x372_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -6.0790882143828848e+42*x375_0
       ))))))
    )))) - x60_4*x67_4 - x7_14 - x82_3 - ((((
       0
    )) if truthy((x230_1)) else ((
       x380_0 + x383_0*((((
          x239_1*x316_0*x318_0*x381_0 - x239_1*x381_0*x382_0
       )) if truthy((x362_0)) else ((
          0
       ))))
    ))))))
    mset(jac, 14, 1, -3.7348863387551538e+22*x365_0 + x95_2*(1.5756322344156688e+19*T*x2_14*x92_2*x93_2*x94_2*x96_2 - vget(X, 0)*x33_7 - 3.7937552985797361e-28*x228_1 - x367_0 - x371_0*x385_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-3.3452431600000003e-24*x372_0 - x373_0*x385_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -1.116213401528895e+46*x375_0
       ))))))
    )))) - x384_0 - x415_0 - ((((
       0
    )) if truthy((x230_1)) else ((
       x380_0 + x383_0*((((
          x316_0*x318_0*x386_0 - x382_0*x386_0
       )) if truthy((x362_0)) else ((
          0
       ))))
    ))))))
    mset(jac, 14, 2, -3.7369204214442467e+22*x365_0 + x95_2*(vget(X, 2)*x435_0*x56_4*x87_3 + vget(X, 6)*x429_0 - x166_1 - 3.7958214423066343e-28*x228_1 - x26_10*x57_4 - x371_0*x440_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-3.3470650363800003e-24*x372_0 - x373_0*x440_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -1.1168213103516679e+46*x375_0
       ))))))
    )))) + x418_0 + x432_0*x437_0 + x432_0*x443_0 + x433_0*x434_0 + x433_0*x441_0 + x438_0*x439_0 + 1.5764903505567533e+19*x444_0 + x446_0 + 2.1533251622400001e-11*x56_4*x85_3*x86_3 - x57_4*x81_3 - ((((
       0
    )) if truthy((x230_1)) else ((
       x380_0 + x383_0*((((
          x316_0*x318_0*x419_0 - x382_0*x419_0
       )) if truthy((x362_0)) else ((
          0
       ))))
    )))) - ((((
       x200_1*(2.1283484790071863*x101_2*x420_0 + 1.7949111316907187*x179_1*x420_0 - 0.019226585526500282*x182_1*x420_0 + 0.025328436022934504*x183_1*x420_0 + 0.26965574024053274*x184_1*x420_0 - 0.53023929521466884*x185_1*x420_0 - 1.249451749011359*x186_1*x420_0 + 0.00057030427583276537*x188_1*x426_0 - 0.006136941893251451*x190_1*x426_0 - 0.010237293323451529*x191_1*x427_0 + 0.023154795695148129*x193_1*x426_0 + 0.12150741535729581*x194_1*x427_0 + 0.048814803971473773*x195_1*x424_0 - 0.033709845761432836*x196_1*x422_0 - 0.63403983120684049*x197_1*x424_0 + 0.81953608629844088*x198_1*x422_0 + 2.5310936376227748*x420_0*x421_0 - 4.9020655078787438*x422_0*x423_0 + 2.8710012490505563*x424_0*x425_0 - 0.50882525384982424*x427_0*x428_0)
    )) if truthy((x201_1)) else ((
       0
    ))))))
    mset(jac, 14, 3, -3.7389545041333399e+22*x365_0 + x95_2*(-3.7978875860335321e-28*x228_1 - x371_0*x448_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-3.3488869127600003e-24*x372_0 - x373_0*x448_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -1.117429219174441e+46*x375_0
       ))))))
    )))) + 1.5773484666978378e+19*x444_0 + x447_0*x88_3 + x456_0))
    mset(jac, 14, 4, -7.4695011950145084e+22*x365_0 + x95_2*(3.151149938820873e+19*T*x2_14*x92_2*x93_2*x94_2*x96_2 - 7.5872348355797369e-28*x228_1 - x371_0*x457_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-6.6902431600000005e-24*x372_0 - x373_0*x457_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -2.2323456674160042e+46*x375_0
       ))))))
    )))) - x459_0))
    mset(jac, 14, 5, -7.4715352777036004e+22*x365_0 + x95_2*(3.1520080549619573e+19*T*x2_14*x92_2*x93_2*x94_2*x96_2 - 7.5893009793066334e-28*x228_1 - x371_0*x460_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-6.6920650363799998e-24*x372_0 - x373_0*x460_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -2.232953576238777e+46*x375_0
       ))))))
    )))) - x458_0))
    mset(jac, 14, 6, -7.4718067601993997e+22*x365_0 + x95_2*(vget(X, 2)*x429_0 - 7.5895767408863695e-28*x228_1 - x371_0*x462_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-6.6923081963799998e-24*x372_0 - x373_0*x462_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -2.2330347118805627e+46*x375_0
       ))))))
    )))) + 3.1521225849724215e+19*x444_0 + x445_0*x463_0 + 3.5888752704000004e-18*x450_0 + 8.7959487653999994e-28*x451_0 + 3.5888752704000004e-18*x452_0 + 2.8278414518999996e-18*x453_0 + x455_0 - x461_0 - x464_0))
    mset(jac, 14, 7, -7.4735693603926949e+22*x365_0 + x95_2*(3.1528661711030424e+19*T*x2_14*x92_2*x93_2*x94_2*x96_2 - 7.5913671230335325e-28*x228_1 - x371_0*x465_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-6.6938869127600005e-24*x372_0 - x373_0*x465_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -2.23356148506155e+46*x375_0
       ))))))
    )))) - x458_0))
    mset(jac, 14, 8, -7.4738408428884933e+22*x365_0 + x95_2*(3.1529807011135066e+19*T*x2_14*x92_2*x93_2*x94_2*x96_2 - vget(X, 0)*x11_14*x9_14 + 5.6556829037999991e-24*vget(X, 2)*vget(X, 3)*x414_0*x466_0*x88_3*x90_2 + 1.7591897530800001e-33*vget(X, 2)*vget(X, 6)*x414_0*x466_0 - vget(X, 2)*x165_1 + 7.1777505407999997e-24*vget(X, 8)*x414_0*x466_0*x86_3*x87_3 + 1.4355501081600001e-11*x103_2*x108_2*x115_2*x408_0*x412_0*x97_2 - x116_2*x435_0 - 7.5916428846132686e-28*x228_1 - x367_0 - x371_0*x467_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-6.6941300727600005e-24*x372_0 - x373_0*x467_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -2.2336426207033358e+46*x375_0
       ))))))
    )))) + 7.1777505407999997e-24*x414_0*x466_0*x83_3*x85_3 - x461_0 - x464_0 + 7.1777505408000004e-12*x56_4*x86_3*x87_3 - ((((
       0
    )) if truthy((x230_1)) else ((
       x364_0 + x380_0 + x383_0*((((
          x239_1*x248_0*x316_0*x318_0 - x249_0*x382_0
       )) if truthy((x362_0)) else ((
          0
       ))))
    ))))))
    mset(jac, 14, 9, -1.1206421616458753e+23*x365_0 + x95_2*(-1.1383056277886369e-27*x228_1 - x371_0*x468_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-1.0037308196379999e-23*x372_0 - x373_0*x468_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -3.3491669777676719e+46*x375_0
       ))))))
    )))) + 4.7276402893776257e+19*x444_0 + x456_0))
    mset(jac, 14, 10, -1.1208455699147847e+23*x365_0 + x95_2*(4.7284984055187104e+19*T*x2_14*x92_2*x93_2*x94_2*x96_2 - 1.1385122421613269e-27*x228_1 - x371_0*x469_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-1.003913007276e-23*x372_0 - x373_0*x469_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -3.3497748865904447e+46*x375_0
       ))))))
    )))) - x415_0 - x458_0 - ((((
       x199_1
    )) if truthy((x201_1)) else ((
       0
    ))))))
    mset(jac, 14, 11, -1.4939002390029017e+23*x365_0 + x95_2*(6.302299877641746e+19*T*x2_14*x92_2*x93_2*x94_2*x96_2 - vget(X, 0)*x35_7 - 8.5199999999999994e-27*vget(X, 0)*x5_14 - 1.5174469671159474e-27*x228_1 - x371_0*x470_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-1.3380486320000001e-23*x372_0 - x373_0*x470_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -4.4646913348320083e+46*x375_0
       ))))))
    )))) - x458_0))
    mset(jac, 14, 12, -1.4941036472718107e+23*x365_0 + x95_2*(6.3031579937828299e+19*T*x2_14*x92_2*x93_2*x94_2*x96_2 - vget(X, 0)*x17_12 - vget(X, 0)*x43_6 - 1.5176535814886368e-27*x228_1 - x371_0*x471_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-1.3382308196379999e-23*x372_0 - x373_0*x471_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -4.4652992436547811e+46*x375_0
       ))))))
    )))) - x459_0 - x472_0*x65_4 - x472_0*x71_3 - x61_4 - x73_3*x75_3))
    mset(jac, 14, 13, -1.4943070555407201e+23*x365_0 + x95_2*(6.3040161099239145e+19*T*x2_14*x92_2*x93_2*x94_2*x96_2 - 1.5178601958613267e-27*x228_1 - x367_0 - x371_0*x473_0 - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       x378_0*(-1.338413007276e-23*x372_0 - x373_0*x473_0 - x377_0*(x374_0 + x376_0*((((
          0
       )) if truthy((x207_1)) else ((
          -4.4659071524775539e+46*x375_0
       ))))))
    )))) - x75_3*x77_3 - ((((
       0
    )) if truthy((x230_1)) else ((
       x380_0 + x383_0*((((
          x316_0*x318_0*x474_0 - x382_0*x474_0
       )) if truthy((x362_0)) else ((
          0
       ))))
    ))))))
    mset(jac, 14, 14, x95_2*(-3.2067318316078082e-16*exp((-1.6499999999999999)*log(abs(T)))*x477_0 - 1.10034915790464e-21*exp((-0.65000000000000002)*log(abs(T)))*x477_0 - vget(X, 0)*x14_14 - 1.0649999999999999e-27*vget(X, 0)*x4_14*x47_4 + 2.185341195413336e-30*vget(X, 1)*x495_0 + 8.741364781653344e-30*vget(X, 11)*x495_0 + 3.12599925e-16*vget(X, 12)*x500_0*x72_3 + vget(X, 2)*vget(X, 3)*x439_0*x88_3*(-0.0064764051000000007*exp((0.04610000000000003)*log(abs(T))) - 2.7293978880000002e-10*exp((2.0424000000000002)*log(abs(T))) - 1.229450816e-13*exp((2.7740999999999998)*log(abs(T))))/((x89_2)*(x89_2)) + vget(X, 3)*x447_0*(1.3296555000000001e-10*exp((-0.90150700000000006)*log(abs(T))) + 2.466314622e-10*exp((-0.44389999999999996)*log(abs(T))) + 8.1647792100000001e-16*exp((1.1825999999999999)*log(abs(T)))) - x118_2*(69500.0*x106_2*x48_4 - x445_0*x505_0) - x118_2*(x409_0*x505_0 + 12307692.307692308*x114_2*x5_14*(0.0042250000000000005*x109_2*x111_2*x488_0 - 4.0625000000000001e-8*x112_2*x38_7 - 0.00048750000000000003*x488_0*exp(-58000.0*x8_14))*exp(x110_2)/x109_2) + 1.5653274417833479e-24*x20_12*x497_0*x72_3 - 0.00090725967999999999*x218_1*x225_1*x304_0*x94_2 - 1.2700000000000001e-21*x23_12*x483_0*x79_3 + 2.62395452e-11*x366_0*x478_0*x59_4 - 5.5399999999999998e-17*x366_0*x58_4*((((
       0
    )) if truthy((x16_12)) else ((
       -0.39700000000000002*exp((-1.397)*log(abs(T)))
    )))) - x37_7*x42_7*((((
       0
    )) if truthy((x16_12)) else ((
       -1.5*x488_0
    )))) - 1.5499999999999999e-26*x37_7*((((
       0
    )) if truthy((x16_12)) else ((
       0.36470000000000002*exp((-0.63529999999999998)*log(abs(T)))
    )))) - x379_0*((((
       0
    )) if truthy((x216_1)) else ((
       -x222_1*x378_0*x481_0*x91_2*x94_2
    )))) + 5.8280000000000003e-8*x40_7*x41_7*x479_0 + x405_0*x56_4*x86_3*(-1.2500000000000001e-32*x38_7 - 1.875e-33*x496_0) - x406_0*(x145_2*(-2.3025850929940459*x131_2*(75.773826*x102_2*x119_2*x8_14 - 14.509090000000008*x512_0 - 13.899501000000001*x515_0 - 2848700.6345267999*x517_0 - 331159.79815649998*x516_0/x133_2) + 4790.3210533157426*x144_2*x48_4 + x388_0*(x389_0*x519_0 + x518_0*log(x141_2)) + x393_0*(x394_0*x519_0 + x518_0*log(x125_2)) + 54584.391438988954*x48_4 - 157.54846734442862*x512_0 + 198.95454259823751*x513_0 - 32.004783802655837*x515_0 - 6559375.6154640894*x517_0) + x164_2*(-2.3025850929940459*x153_2*(70.138370000000009*x101_2*x8_14 - 9.4070299999999989*x513_0 - 0.77462909999999996*x515_0 - 588180.10479140002*x520_0 - 160821.97128249999*x516_0/x154_2) + 3816.3275589792611*x163_2*x48_4 + x397_0*(x398_0*x522_0 + x521_0*log(x160_2)) + x402_0*(x403_0*x522_0 + x521_0*log(x149_2)) + 49431.413233526648*x48_4 + 98.337445626384849*x512_0 - 9.3363608541157479*x513_0 - 1.783649418259394*x515_0 - 1354334.7412883535*x520_0)) + 0.00084373771595996178*x416_0*x93_2 + 7.1777505407999997e-24*x434_0*x503_0 + x437_0*x503_0 + 7.1777505407999997e-24*x441_0*x503_0 + x443_0*x503_0 + 3.4968000000000002e-9*x479_0*exp(-564000.0*x24_11) - x480_0*x484_0 - x480_0*x487_0 - x484_0*x485_0 - x485_0*x487_0 + 2.9662164452379397e-24*x486_0*x491_0*x501_0 - x489_0*x74_3 - x489_0*x78_3 + 2.3717082451262844e-21*x490_0*x492_0 + 8.8760999999999989e-14*x490_0*x493_0 + 4.0160926284138423e-24*x492_0*x502_0 + 2.0041755700000002e-16*x493_0*x502_0 - 5.0099999999999997e-27*x494_0*x63_4 - 9.1000000000000001e-27*x494_0*x69_3 + 1.7519018237332822e-19*x497_0*x59_4 + 1.5843011077443579e-29*x498_0*x64_4 + 2.8776726707532255e-29*x498_0*x70_3 + 2.7724337999999999e-22*x499_0*x64_4 + 1.199289e-22*x499_0*x70_3 + 2.6764460520000001e-16*x500_0*x501_0 + 7.1777505408000004e-12*x56_4*x83_3*(-1.0000000000000001e-31*x38_7 - 1.5e-32*x496_0) - ((((
       0
    )) if truthy((x230_1)) else ((
       x383_0*((((
          x299_0*x318_0*x530_0 + x319_0*x533_0 + x382_0*(-x530_0 - x533_0)
       )) if truthy((x362_0)) else ((
          0
       ))))
    )))) - ((((
       x200_1*(50.504556041967454*x101_2*x506_0 + 0.011577397847574064*x168_1*x192_1*x508_0 + 0.00057030427583276537*x171_1*x187_1*x508_0 - 0.0046027064199385889*x172_1*x189_1*x508_0 - 46.931151210299063*x179_1*x506_0 - 0.0084274614403582089*x181_1*x506_0 + 0.27317869543281359*x183_1*x506_0 - 1.5965204000783517*x184_1*x506_0 - 2.4510327539393719*x185_1*x506_0 + 15.190568323798459*x186_1*x506_0 - 0.013649724431268705*x190_1*x510_0 + 0.12150741535729581*x193_1*x510_0 + 0.097629607942947547*x194_1*x511_0 - 0.33921683589988288*x196_1*x507_0 - 0.076906342106001127*x197_1*x509_0 + 2.8710012490505563*x198_1*x507_0 + 1.7949111316907187*x421_0*x506_0 - 2.4989034980227181*x423_0*x507_0 + 0.80896722072159821*x425_0*x509_0 - 0.95105974681026062*x428_0*x511_0)
    )) if truthy((x201_1)) else ((
       0
    )))))/(vget(X, 0)*x475_0 + vget(X, 1)*x475_0 + vget(X, 10)*x476_0 + vget(X, 11)*x475_0 + vget(X, 12)*x475_0 + vget(X, 13)*x475_0 + vget(X, 2)*x475_0 + vget(X, 3)*x475_0 + vget(X, 4)*x475_0 + vget(X, 5)*x475_0 + vget(X, 6)*x476_0 + vget(X, 7)*x475_0 + vget(X, 8)*x476_0 + vget(X, 9)*x476_0))


def actual_jac(state: BurnState, mut jac: MatTensor):
    var z = redshift()
    var X = SpeciesTensor.stack_allocation()
    for i in range(NumSpec):
        vset(X, i, state.xn[i])
    jac_nuc(state, jac, X, z)


def burn_state_from_y(y: VecTensor) -> BurnState:
    var state = BurnState()
    for n in range(NumSpec):
        state.xn[n] = max(vget(y, n), small_number_density_floor())
    state.e = vget(y, NetIenuc)
    eos_re(state)
    return state^


def zero_vec(mut v: VecTensor):
    for i in range(Neqs):
        vset(v, i, 0.0)


def zero_mat(mut m: MatTensor):
    for i in range(Neqs):
        for j in range(Neqs):
            mset(m, i, j, 0.0)


def copy_vec(src: VecTensor, mut dst: VecTensor):
    for i in range(Neqs):
        vset(dst, i, vget(src, i))


def lu_decomposition(mut A: MatTensor, mut ipvt: InlineArray[Int, Neqs]) -> Int:
    var info = 0
    for k in range(Neqs - 1):
        var pivot_row = k
        var max_val = abs(mget(A, k, k))
        for i in range(k + 1, Neqs):
            var value = abs(mget(A, i, k))
            if value > max_val:
                max_val = value
                pivot_row = i
        ipvt[k] = pivot_row
        if mget(A, pivot_row, k) != 0.0:
            if pivot_row != k:
                var t = mget(A, pivot_row, k)
                mset(A, pivot_row, k, mget(A, k, k))
                mset(A, k, k, t)
                for j in range(k + 1, Neqs):
                    var trailing = mget(A, pivot_row, j)
                    mset(A, pivot_row, j, mget(A, k, j))
                    mset(A, k, j, trailing)
            var multiplier = -1.0 / mget(A, k, k)
            for i in range(k + 1, Neqs):
                mset(A, i, k, mget(A, i, k) * multiplier)
            for j in range(k + 1, Neqs):
                var t = mget(A, k, j)
                for i in range(k + 1, Neqs):
                    mset(A, i, j, mget(A, i, j) + t * mget(A, i, k))
        else:
            info = k + 1
    ipvt[Neqs - 1] = Neqs - 1
    if mget(A, Neqs - 1, Neqs - 1) == 0.0:
        info = Neqs
    return info


def lu_solve(LU: MatTensor, ipvt: InlineArray[Int, Neqs], mut x: VecTensor):
    for k in range(Neqs - 1):
        var pivot_row = ipvt[k]
        var t = vget(x, pivot_row)
        if pivot_row != k:
            vset(x, pivot_row, vget(x, k))
            vset(x, k, t)
        for j in range(k + 1, Neqs):
            vset(x, j, vget(x, j) + t * mget(LU, j, k))
    for kb in range(Neqs):
        var k = Neqs - 1 - kb
        vset(x, k, vget(x, k) / mget(LU, k, k))
        var t = -vget(x, k)
        for j in range(k):
            vset(x, j, vget(x, j) + t * mget(LU, j, k))


def configure_ros2s(mut rtol_vec: VecTensor, mut atol_vec: VecTensor):
    for n in range(NumSpec):
        vset(rtol_vec, n, RtolSpec)
        vset(atol_vec, n, AtolSpec)
    vset(rtol_vec, NetIenuc, RtolEnergy)
    vset(atol_vec, NetIenuc, AtolEnergy)


def eval_jacobian(y: VecTensor, mut fjac: MatTensor, mut stats: IntegratorStats):
    zero_mat(fjac)
    var state = burn_state_from_y(y)
    actual_jac(state, fjac)
    stats.jacobian_calls += 1


def decompose(fjac: MatTensor, mut e: MatTensor, mut ip: InlineArray[Int, Neqs], fac: Float64, mut stats: IntegratorStats) -> Int:
    for i in range(Neqs):
        for j in range(Neqs):
            mset(e, i, j, -mget(fjac, i, j))
        mset(e, i, i, mget(e, i, i) + fac)
    var info = lu_decomposition(e, ip)
    if info == 0:
        stats.decompositions += 1
    return info


def solve(e: MatTensor, ip: InlineArray[Int, Neqs], mut ak: VecTensor, mut stats: IntegratorStats):
    lu_solve(e, ip, ak)
    stats.linear_solves += 1


def rhs_from_y(t: Float64, y: VecTensor, mut out: VecTensor):
    var state = burn_state_from_y(y)
    actual_rhs(state, out)


def error_norm(y: VecTensor, ynew: VecTensor, work: VecTensor, rtol_vec: VecTensor, atol_vec: VecTensor) -> Float64:
    var err = 0.0
    for i in range(Neqs):
        var sk = vget(atol_vec, i) + vget(rtol_vec, i) * max(abs(vget(y, i)), abs(vget(ynew, i)))
        var term = vget(work, i) / sk
        err += term * term
    return sqrt(err / Float64(Neqs))


def integrate_ros2s(mut y: VecTensor, rtol_vec: VecTensor, atol_vec: VecTensor, dt: Float64, mut stats: IntegratorStats) -> Int:
    comptime gamma = 0.292893218813452
    comptime ct2 = 0.585786437626905
    comptime a21 = 2.0000000000000036
    comptime a31 = 6.828427124746214
    comptime a32 = 3.4142135623731007
    comptime c21 = -6.828427124746214
    comptime c31 = -10.949747468305889
    comptime c32 = -7.535533905932761
    comptime b1 = 6.828427124746214
    comptime b2 = 3.414213562373101
    comptime b3 = 1.0
    comptime e1 = -0.23570226039551292
    comptime e2 = -0.23570226039551567
    comptime e3 = -0.13807118745769906

    var ynew = VecTensor.stack_allocation()
    var ak1 = VecTensor.stack_allocation()
    var ak2 = VecTensor.stack_allocation()
    var work = VecTensor.stack_allocation()
    var fjac = MatTensor.stack_allocation()
    var e = MatTensor.stack_allocation()
    var dy = VecTensor.stack_allocation()
    var ip = InlineArray[Int, Neqs](fill=0)

    var tout = dt
    var uround = 1.0e-16
    var fac_min = 0.2
    var fac_max = 6.0
    var safe = 0.9
    var max_steps = 10000000
    if tout <= 0.0 or safe <= 0.001 or safe >= 1.0 or fac_min <= 0.0 or fac_max < 1.0:
        return BadInputs
    for i in range(Neqs):
        if vget(atol_vec, i) <= 0.0 or vget(rtol_vec, i) <= 10.0 * uround:
            return TooMuchAccuracyRequested

    var hmaxn = tout
    var h = dt
    if abs(h) <= 10.0 * uround:
        h = 1.0e-6
    h = min(abs(h), hmaxn)

    var reject = False
    var last = False
    var nsing = 0
    var hacc = 0.0
    var erracc = 1.0
    var x = 0.0
    var n_step = 0
    var n_accept = 0

    while True:
        if n_step > max_steps:
            return TooManySteps
        if 0.1 * abs(h) <= abs(x) * uround:
            return DtUnderflow
        if last:
            return Success

        if x + h * 1.0001 >= tout:
            h = tout - x
            last = True

        eval_jacobian(y, fjac, stats)

        while True:
            var fac = 1.0 / (h * gamma)
            if decompose(fjac, e, ip, fac, stats) != 0:
                nsing += 1
                if nsing >= 5:
                    return LuDecompositionError
                h *= 0.5
                reject = True
                last = False
                continue

            rhs_from_y(x, y, ak1)
            solve(e, ip, ak1, stats)

            for i in range(Neqs):
                vset(ynew, i, vget(y, i) + a21 * vget(ak1, i))
                vset(ak2, i, (c21 / h) * vget(ak1, i))
            rhs_from_y(x + ct2 * h, ynew, dy)
            for i in range(Neqs):
                vset(ak2, i, vget(ak2, i) + vget(dy, i))
            solve(e, ip, ak2, stats)

            for i in range(Neqs):
                vset(ynew, i, vget(y, i) + a31 * vget(ak1, i) + a32 * vget(ak2, i))
                vset(work, i, (c31 * vget(ak1, i) + c32 * vget(ak2, i)) / h)
            rhs_from_y(x + h, ynew, dy)
            for i in range(Neqs):
                vset(work, i, vget(work, i) + vget(dy, i))
            solve(e, ip, work, stats)

            for i in range(Neqs):
                var ak3i = vget(work, i)
                vset(ynew, i, vget(y, i) + b1 * vget(ak1, i) + b2 * vget(ak2, i) + b3 * ak3i)
                vset(work, i, e1 * vget(ak1, i) + e2 * vget(ak2, i) + e3 * ak3i)
            stats.rhs_calls += 3
            n_step += 1
            stats.internal_steps += 1

            var err = error_norm(y, ynew, work, rtol_vec, atol_vec)
            var raw_fac = cbrt(err) / safe
            var lower_fac = 1.0 / fac_max
            var upper_fac = 1.0 / fac_min
            var fac_step = max(lower_fac, min(upper_fac, raw_fac))
            var hnew = h / fac_step
            if err <= 1.0:
                n_accept += 1
                stats.accepted_steps += 1
                if n_accept > 1:
                    var facgus = max(
                        1.0 / fac_max,
                        min(1.0 / fac_min, (hacc / h) * cbrt((err * err) / erracc) / safe),
                    )
                    hnew = h / max(fac_step, facgus)
                hacc = h
                erracc = max(1.0e-2, err)
                copy_vec(ynew, y)
                x += h
                if abs(hnew) > hmaxn:
                    hnew = hmaxn
                if reject:
                    hnew = min(abs(hnew), abs(h))
                reject = False
                h = hnew
                break

            reject = True
            last = False
            h = hnew
            if n_accept >= 1:
                stats.rejected_steps += 1


def burn_ros2s(mut state: BurnState, dt: Float64, mut stats: IntegratorStats) -> Int:
    eos_rt(state)
    var y = VecTensor.stack_allocation()
    var rtol_vec = VecTensor.stack_allocation()
    var atol_vec = VecTensor.stack_allocation()
    configure_ros2s(rtol_vec, atol_vec)
    for n in range(NumSpec):
        vset(y, n, state.xn[n])
    vset(y, NetIenuc, state.e)

    var result = integrate_ros2s(y, rtol_vec, atol_vec, dt, stats)
    if result != Success:
        return result

    for n in range(NumSpec):
        state.xn[n] = vget(y, n)
    state.e = vget(y, NetIenuc)
    return result


def initial_number_density(n: Int) -> Float64:
    if n == 0:
        return 1.0e-4
    if n == 1:
        return 1.0e-4
    if n == 2:
        return 1.0e0
    if n == 8:
        return 1.0e-6
    if n == 13:
        return 0.0775
    return 1.0e-40


def make_initial_state() -> BurnState:
    var state = BurnState()
    state.T = InitialTemperature
    for n in range(NumSpec):
        state.xn[n] = initial_number_density(n)
    state.rho = density(state.xn)
    normalize_number_densities_to_density(state)
    eos_rt(state)
    return state^


def make_collapse_state() -> CollapseState:
    var collapse = CollapseState()
    collapse.current = make_initial_state()
    collapse.density_driver = collapse.current.rho
    return collapse^


def splitmix64(value_in: UInt64) -> UInt64:
    var value = value_in + UInt64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> 27)) * UInt64(0x94D049BB133111EB)
    return value ^ (value >> 31)


def perturbation_factor(cell: Int, step: Int) -> Float64:
    var seed = (UInt64(cell) << 32) ^ (UInt64(step) << 16)
    var bits = splitmix64(seed) >> 11
    var unit = Float64(bits) * (1.0 / Float64(UInt64(1) << 53))
    return 1.0 + PerturbationAmplitude * (2.0 * unit - 1.0)


def apply_perturbation(mut collapse: CollapseState, cell: Int, step: Int, enabled: Bool):
    if (not enabled) or step == 0 or step % PerturbationInterval != 0:
        return
    var factor = perturbation_factor(cell, step)
    collapse.density_driver *= factor
    collapse.current.rho *= factor
    for n in range(NumSpec):
        collapse.current.xn[n] *= factor
    floor_and_normalize_number_densities(collapse.current)
    balance_charge(collapse.current)
    floor_and_normalize_number_densities(collapse.current)
    eos_re(collapse.current)


def advance_collapse_step(mut collapse: CollapseState, cell: Int, step: Int, perturb: Bool) -> Int:
    apply_perturbation(collapse, cell, step, perturb)
    var old_density = collapse.density_driver
    var rho = density(collapse.current.xn)
    var tff = sqrt(Pi * 3.0 / (32.0 * rho * GravConstant))
    var dt = TffReduc * tff

    collapse.density_driver += dt * (collapse.density_driver / tff)
    if dt < 10.0 or collapse.density_driver > 2.0e-6:
        return Success

    var density_ratio = collapse.density_driver / old_density
    for n in range(NumSpec):
        collapse.current.xn[n] *= density_ratio
    collapse.current.rho *= density_ratio

    var result = burn_ros2s(collapse.current, dt, collapse.stats)
    if result != Success:
        print("ROS2S failed on collapse step", step, "cell", cell, "with code", result)
        return result

    floor_and_normalize_number_densities(collapse.current)
    balance_charge(collapse.current)
    floor_and_normalize_number_densities(collapse.current)
    eos_re(collapse.current)

    collapse.time += dt
    collapse.completed_steps += 1
    return 0


def add_stats(mut total: IntegratorStats, value: IntegratorStats):
    total.internal_steps += value.internal_steps
    total.rhs_calls += value.rhs_calls
    total.jacobian_calls += value.jacobian_calls
    total.decompositions += value.decompositions
    total.linear_solves += value.linear_solves
    total.accepted_steps += value.accepted_steps
    total.rejected_steps += value.rejected_steps


def checked_cell_count(grid_dim: Int) raises -> Int:
    var cells = grid_dim * grid_dim * grid_dim
    if grid_dim <= 0 or cells <= 0:
        raise Error("grid dimension must be positive")
    return cells


def print_usage(program: String):
    print("usage:", program, "[--grid N] [--perturb|--no-perturb] [--no-compare-final-state]")


def parse_args() raises -> Options:
    var args = argv()
    var options = Options()
    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--help" or arg == "-h":
            options.show_help = True
            return options^
        if arg == "--grid":
            if i + 1 >= len(args):
                print_usage(String(args[0]))
                raise Error("missing grid value")
            i += 1
            options.grid_dim = Int(String(args[i]))
            if options.grid_dim <= 0:
                raise Error("grid dimension must be positive")
            i += 1
            continue
        if arg == "--perturb":
            options.perturb = True
            i += 1
            continue
        if arg == "--no-perturb":
            options.perturb = False
            i += 1
            continue
        if arg == "--no-compare-final-state":
            i += 1
            continue
        if arg == "--compare-final-state":
            raise Error("--compare-final-state is not implemented in the Mojo port yet")
        print_usage(String(args[0]))
        raise Error("unrecognized argument")
    return options^


def print_state(state: BurnState):
    print("rho:", state.rho)
    print("T:", state.T)
    print("Eint:", state.e)
    print("E:", state.xn[0])
    print("Hp:", state.xn[1])
    print("H:", state.xn[2])
    print("Hm:", state.xn[3])
    print("Dp:", state.xn[4])
    print("D:", state.xn[5])
    print("H2p:", state.xn[6])
    print("Dm:", state.xn[7])
    print("H2:", state.xn[8])
    print("HDp:", state.xn[9])
    print("HD:", state.xn[10])
    print("HEpp:", state.xn[11])
    print("HEp:", state.xn[12])
    print("HE:", state.xn[13])


def main() raises:
    var options = parse_args()
    var args = argv()
    if options.show_help:
        print_usage(String(args[0]))
        return
    var num_cells = checked_cell_count(options.grid_dim)
    var cells = List[CollapseState]()
    for _ in range(num_cells):
        cells.append(make_collapse_state())

    var failure = Success
    var completed_global_steps = 0
    for step in range(MaxCollapseSteps):
        var all_stopped = True
        for cell in range(num_cells):
            if cells[cell].completed_steps < completed_global_steps:
                continue
            var result = advance_collapse_step(cells[cell], cell, step, options.perturb)
            if result != Success and result != 0:
                failure = result
                break
            var stopped = result == Success
            all_stopped = all_stopped and stopped
        if failure != Success or all_stopped:
            break
        completed_global_steps += 1

    if failure != Success:
        raise Error("integration failed")

    var total_stats = IntegratorStats()
    for cell in range(num_cells):
        add_stats(total_stats, cells[cell].stats)

    print("Primordial chemistry collapse grid with ROS2S")
    print("grid:", options.grid_dim, "^3 (", num_cells, "cells )")
    print("perturbations:", "enabled" if options.perturb else "disabled")
    print("completed global collapse steps:", completed_global_steps)
    if num_cells == 1:
        print("representative cell completed steps:", cells[0].completed_steps)
        print("representative cell physical time:", cells[0].time)
        print("representative cell density driver:", cells[0].density_driver)
    else:
        print("cell summaries for multi-cell runs are not implemented in the Mojo port yet")
    print("ROS2S internal steps:", total_stats.internal_steps)
    print("ROS2S rhs calls:", total_stats.rhs_calls)
    print("ROS2S jacobian calls:", total_stats.jacobian_calls)
    print("ROS2S decompositions:", total_stats.decompositions)
    print("ROS2S linear solves:", total_stats.linear_solves)
    print("ROS2S accepted/rejected:", total_stats.accepted_steps, "/", total_stats.rejected_steps)
    if options.grid_dim == 1:
        print_state(cells[0].current)
