// Largest Triangle  -  ETF SA Contests / Day 1
// -----------------------------------------------------------------------------
// Given N integer points, find the maximum-area triangle and print 2*area
// (an integer).
//
// Two facts make this efficient:
//   1. The three vertices of the maximum-area triangle are always vertices of
//      the convex hull of the point set.  So we only need the hull.
//   2. On a convex polygon, for a fixed first vertex `a`, as the second vertex
//      `b` advances, the apex `c` that maximises area(a,b,c) is monotonically
//      non-decreasing.  This gives an O(h^2) two-pointer scan over the hull
//      (h = number of hull vertices).
//
// A convex lattice polygon whose coordinates lie in [0, L] has only O(L^(2/3))
// vertices, so h stays small (a few thousand at most) regardless of N.
//
// All arithmetic fits comfortably in 64-bit integers:
//   coordinates <= 2e5  ->  doubled area <= 4e10  <  9.2e18.
//
// Complexity: O(N log N) for the hull + O(h^2) for the scan.
// -----------------------------------------------------------------------------
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct P { ll x, y; };

// Cross product of OA x OB.  > 0 => left turn (counter-clockwise).
static inline ll cross(const P& O, const P& A, const P& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<P> pts(n);
    for (int i = 0; i < n; ++i) cin >> pts[i].x >> pts[i].y;

    // --- Convex hull (Andrew's monotone chain) -------------------------------
    sort(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    pts.erase(unique(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.x == b.x && a.y == b.y;
    }), pts.end());

    int m = (int)pts.size();
    if (m < 3) { cout << 0 << '\n'; return 0; }   // all points coincide / are 2

    vector<P> h(2 * m);
    int k = 0;
    for (int i = 0; i < m; ++i) {                 // lower hull
        while (k >= 2 && cross(h[k - 2], h[k - 1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    for (int i = m - 2, t = k + 1; i >= 0; --i) { // upper hull
        while (k >= t && cross(h[k - 2], h[k - 1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    h.resize(k - 1);

    int hs = (int)h.size();
    if (hs < 3) { cout << 0 << '\n'; return 0; }  // all points collinear

    auto area2 = [&](int a, int b, int c) -> ll {
        return llabs(cross(h[a], h[b], h[c]));
    };

    // --- Maximum-area triangle on the hull (O(h^2) two-pointer) --------------
    ll best = 0;
    for (int a = 0; a < hs; ++a) {
        int c = a + 2;
        for (int b = a + 1; b < hs; ++b) {
            if (c <= b) c = b + 1;
            if (c >= hs) break;
            // area2(a,b,c) is unimodal in c; advance c while it keeps growing.
            while (c + 1 < hs && area2(a, b, c + 1) >= area2(a, b, c)) ++c;
            best = max(best, area2(a, b, c));
        }
    }

    cout << best << '\n';
    return 0;
}
