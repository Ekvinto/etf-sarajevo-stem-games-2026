// Gaussian Blur  -  ETF SA Contests / Day 1
// -----------------------------------------------------------------------------
// For a query centre (i0,j0) and fixed radius R, constant C, we must output
//     sum over |i|,|j| <= R of  A[i0+i][j0+j] * C^(|i|+|j|)   (mod 1e9+7)
//
// The weight C^(|i|+|j|) = C^|i| * C^|j| is SEPARABLE, and R,C are the same for
// every query, so the whole answer is a single 2-D convolution of A with a
// fixed separable kernel.  We precompute it once; every query is then an O(1)
// table lookup.
//
// 1-D weighted window:  W(c) = sum_{j=-R..R} A[c+j] * C^|j|  (out-of-range = 0)
// Split into a right part and a left part, each obeying a linear recurrence:
//
//   Right(c) = sum_{j=0..R}  A[c+j]*C^j
//            = A[c] + C*Right(c+1) - C^(R+1)*A[c+R+1]
//
//   Left(c)  = sum_{j=1..R}  A[c-j]*C^j
//   Left(c+1)= C*A[c] + C*Left(c) - C^(R+1)*A[c-R]
//
//   W(c) = Right(c) + Left(c)
//
// We apply this along rows, then along columns of the row-blurred matrix.
// Treating out-of-range cells as 0 makes the recurrence yield the correctly
// clipped sum for every cell; interior cells (the only ones queried) get the
// exact full-window sum.
//
// Complexity: O(M*N) preprocessing + O(T) for the queries.
// Everything is done modulo p = 1e9+7; long long avoids overflow.
// -----------------------------------------------------------------------------
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const ll MOD = 1000000007LL;

// Apply the 1-D weighted window (radius R, weight C^|.|) to one line `src`
// of length `len`, writing the result into `dst`.  cR1 = C^(R+1) mod MOD.
static void blurLine(const vector<ll>& src, vector<ll>& dst,
                     int len, int R, ll C, ll cR1) {
    static thread_local vector<ll> Right, Left;
    Right.assign(len + 1, 0);
    Left.assign(len + 1, 0);

    // Right(c) computed from right to left.
    for (int c = len - 1; c >= 0; --c) {
        ll term = (c + R + 1 < len) ? src[c + R + 1] : 0;
        ll v = src[c] + C * Right[c + 1] % MOD - cR1 * term % MOD;
        Right[c] = (v % MOD + MOD) % MOD;
    }
    // Left(c+1) computed from left to right.
    for (int c = 0; c < len; ++c) {
        ll term = (c - R >= 0) ? src[c - R] : 0;
        ll v = C * src[c] % MOD + C * Left[c] % MOD - cR1 * term % MOD;
        Left[c + 1] = (v % MOD + MOD) % MOD;
    }
    for (int c = 0; c < len; ++c)
        dst[c] = (Right[c] + Left[c]) % MOD;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int M, N, R;
    ll C;
    if (!(cin >> M >> N >> R >> C)) return 0;
    C %= MOD;

    vector<vector<ll>> A(M, vector<ll>(N));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            cin >> A[i][j];

    // C^(R+1) mod MOD.
    ll cR1 = 1;
    for (int e = 0; e < R + 1; ++e) cR1 = cR1 * C % MOD;

    // --- Row pass: B[r][c] = weighted window over columns --------------------
    vector<vector<ll>> B(M, vector<ll>(N));
    for (int r = 0; r < M; ++r)
        blurLine(A[r], B[r], N, R, C, cR1);

    // --- Column pass on B: Ans[r][c] = weighted window over rows -------------
    vector<vector<ll>> Ans(M, vector<ll>(N));
    {
        vector<ll> col(M), res(M);
        for (int c = 0; c < N; ++c) {
            for (int r = 0; r < M; ++r) col[r] = B[r][c];
            blurLine(col, res, M, R, C, cR1);
            for (int r = 0; r < M; ++r) Ans[r][c] = res[r];
        }
    }

    // --- Answer the queries --------------------------------------------------
    int T;
    cin >> T;
    string out;
    out.reserve((size_t)T * 12);
    for (int t = 0; t < T; ++t) {
        int i0, j0;
        cin >> i0 >> j0;
        out += to_string(Ans[i0][j0]);
        out += '\n';
    }
    cout << out;
    return 0;
}
