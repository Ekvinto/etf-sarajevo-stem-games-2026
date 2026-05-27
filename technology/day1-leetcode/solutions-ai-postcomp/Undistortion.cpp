// Undistortion  -  ETF SA Contests / Day 1
// -----------------------------------------------------------------------------
// We are given the distortion parameters k1, k2 and a DISTORTED point pd(xd,yd).
// The forward model is:
//     r2 = xu*xu + yu*yu
//     s  = 1 + r2*k1 + r2*r2*k2
//     xd = xu*s ,  yd = yu*s
//
// Key observation: pd = s * pu, so pd and pu share the same direction from the
// origin and only differ in length.  With ru = |pu| and rd = |pd|:
//     rd = ru * s = ru + k1*ru^3 + k2*ru^5  =: f(ru)
//
// Because k1,k2 >= 0 and ru >= 0, f is strictly increasing on [0, +inf), and
// since s >= 1 we have ru <= rd.  So we binary-search ru in [0, rd].
// Once ru is known, pu = pd * (ru / rd)  (i.e. divide by s).
//
// Complexity: O(T * iterations).  Trivially within the limits.
// -----------------------------------------------------------------------------
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;

    cout << fixed << setprecision(3);

    for (int t = 0; t < T; ++t) {
        double k1, k2, xd, yd;
        cin >> k1 >> k2 >> xd >> yd;

        double rd = sqrt(xd * xd + yd * yd);
        double xu, yu;

        if (rd < 1e-15) {
            // Distorted point is the origin -> undistorted point is the origin.
            xu = 0.0;
            yu = 0.0;
        } else {
            // Solve f(ru) = rd for ru in [0, rd] by bisection.
            double lo = 0.0, hi = rd;
            for (int it = 0; it < 200; ++it) {
                double mid = 0.5 * (lo + hi);
                double r2  = mid * mid;
                double f   = mid * (1.0 + r2 * k1 + r2 * r2 * k2);
                if (f < rd) lo = mid;
                else        hi = mid;
            }
            double ru    = 0.5 * (lo + hi);
            double scale = ru / rd;        // == 1 / s
            xu = xd * scale;
            yu = yd * scale;
        }

        // Avoid printing "-0.000" for values that round to zero.
        if (fabs(xu) < 5e-4) xu = 0.0;
        if (fabs(yu) < 5e-4) yu = 0.0;

        cout << xu << ' ' << yu << '\n';
    }
    return 0;
}
