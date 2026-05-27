// Find the Square  -  Robust Heuristic Vision Solution
// -----------------------------------------------------------------------------
// PROBLEM:
// We are given an M x N grayscale image (0-255) containing a single bright
// "white" square. The square may be tilted, viewed in perspective (skewed into
// a quadrilateral), or placed on highly noisy/complex backgrounds. We must
// output the 4 corner coordinates, tolerating a +/- 1 pixel error.
//
// APPROACH:
// Because lighting and background complexity vary wildly, a single global
// threshold (like Otsu's) fails when the background is bright (e.g., a sky).
// Instead, we use a Multi-Thresholding Heuristic Pipeline:
//
//   1. THRESHOLD SWEEPING: We evaluate the image at multiple strict brightness
//      thresholds (from 250 down to 50). At each level, we extract Connected
//      Components (8-connectivity) to find candidate shapes.
//   2. CONVEX HULL: For each component > 10 pixels, we compute its convex hull.
//      This ignores internal noise (holes) and simplifies the outer boundary.
//   3. QUADRILATERAL REDUCTION: We iteratively remove the vertex that forms
//      the smallest triangle with its neighbors. This "irons out" jagged pixel
//      edges, safely reducing any square-like shape into exactly 4 corners.
//
// SCORING METRICS (The Secret Sauce):
// To separate the true square from distractor circles, skies, or rectangles,
// each candidate is heavily penalized if it fails these geometric tests:
//
//   * FILL (Pick's Theorem): Calculates the theoretical expected pixel count
//     based on area and perimeter (Area = I + B/2 - 1). Penalizes hollow
//     shapes or sprawling L-shapes.
//   * QUADITY: Ratio of the 4-corner quad area to the original convex hull area.
//     A true quad is ~1.0; a circle drops to ~0.63.
//   * EDGE STRAIGHTNESS: The average distance from the raw hull points to the
//     simplified 4-point boundary. Heavily penalizes curved/blobby edges.
//   * ASPECT RATIO: Ratio of the shortest side to the longest side. Penalizes
//     highly stretched, non-square rectangles.
//
// The candidate across all thresholds with the highest combined score
// (Intensity * Fill * Quadity * Straightness * Aspect) is chosen as the square.
// -----------------------------------------------------------------------------#include <bits/stdc++.h>
using namespace std;

struct Pt { double x, y; };

static inline double crossPt(const Pt& O, const Pt& A, const Pt& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

static double polygonArea(const vector<Pt>& p) {
    double area = 0;
    int n = p.size();
    for (int i = 0; i < n; ++i) {
        area += p[i].x * p[(i + 1) % n].y - p[(i + 1) % n].x * p[i].y;
    }
    return abs(area) / 2.0;
}

static vector<Pt> convexHull(vector<Pt> p) {
    sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    p.erase(unique(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        return a.x == b.x && a.y == b.y;
    }), p.end());

    int n = (int)p.size();
    if (n < 3) return p;

    vector<Pt> h(2 * n);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && crossPt(h[k - 2], h[k - 1], p[i]) <= 0) --k;
        h[k++] = p[i];
    }
    for (int i = n - 2, t = k + 1; i >= 0; --i) {
        while (k >= t && crossPt(h[k - 2], h[k - 1], p[i]) <= 0) --k;
        h[k++] = p[i];
    }
    h.resize(k - 1);
    return h;
}

static vector<Pt> reduceToQuad(vector<Pt> p) {
    while (p.size() > 4) {
        int n = p.size();
        double min_area = 1e18;
        int min_idx = -1;
        for (int i = 0; i < n; ++i) {
            int prev = (i - 1 + n) % n;
            int next = (i + 1) % n;
            double area = abs(crossPt(p[prev], p[i], p[next])) / 2.0;
            if (area < min_area) {
                min_area = area;
                min_idx = i;
            }
        }
        p.erase(p.begin() + min_idx);
    }
    return p;
}

// Distance from point p to line segment a-b
static double distToSegment(Pt p, Pt a, Pt b) {
    double l2 = (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
    if (l2 == 0) return hypot(p.x - a.x, p.y - a.y);
    double t = max(0.0, min(1.0, ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2));
    double proj_x = a.x + t * (b.x - a.x);
    double proj_y = a.y + t * (b.y - a.y);
    return hypot(p.x - proj_x, p.y - proj_y);
}

int M, N;
vector<int> img;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> M >> N)) return 0;
    int total = M * N;
    img.resize(total);
    for (auto& v : img) cin >> v;

    // Sweep thresholds from strict white down to mid-gray
    vector<int> thresholds;
    for (int t = 250; t >= 50; t -= 20) thresholds.push_back(t);

    double best_score = -1.0;
    Pt best_corners[4] = {{0,0}, {(double)(N-1),0}, {(double)(N-1),(double)(M-1)}, {0,(double)(M-1)}};

    static const int dr[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
    static const int dc[8] = {-1, 0, 1,-1, 1,-1, 0, 1};

    for (int thr : thresholds) {
        vector<char> fg(total, 0), visited(total, 0);
        for (int i = 0; i < total; ++i) fg[i] = (img[i] >= thr) ? 1 : 0;

        vector<int> stack;
        vector<Pt> comp;
        stack.reserve(total);

        for (int start = 0; start < total; ++start) {
            if (!fg[start] || visited[start]) continue;

            comp.clear();
            stack.clear();
            stack.push_back(start);
            visited[start] = 1;

            long long sum_intensity = 0;

            while (!stack.empty()) {
                int idx = stack.back(); stack.pop_back();
                int r = idx / N, c = idx % N;
                comp.push_back({(double)c, (double)r});
                sum_intensity += img[idx];

                for (int k = 0; k < 8; ++k) {
                    int nr = r + dr[k], nc = c + dc[k];
                    if (nr < 0 || nr >= M || nc < 0 || nc >= N) continue;
                    int nidx = nr * N + nc;
                    if (fg[nidx] && !visited[nidx]) {
                        visited[nidx] = 1;
                        stack.push_back(nidx);
                    }
                }
            }

            long long count = comp.size();
            if (count < 10) continue;

            vector<Pt> hull = convexHull(comp);
            if (hull.size() < 4) continue;

            double hull_area = polygonArea(hull);
            if (hull_area < 5.0) continue;

            // 1. Pick's Theorem Fill Check
            double hull_perim = 0;
            for (size_t i = 0; i < hull.size(); ++i) {
                double dx = hull[i].x - hull[(i+1)%hull.size()].x;
                double dy = hull[i].y - hull[(i+1)%hull.size()].y;
                hull_perim += sqrt(dx*dx + dy*dy);
            }
            double expected_count = hull_area + hull_perim / 2.0 + 1.0;
            double fill = (double)count / expected_count;
            double fill_penalty = exp(-5.0 * max(0.0, abs(1.0 - fill) - 0.15)); // 15% leniency for jagged grid

            // 2. Reduce to quad and check Quadity
            vector<Pt> quad = reduceToQuad(hull);
            double quad_area = polygonArea(quad);
            double quadity = quad_area / hull_area;
            double quadity_penalty = exp(-5.0 * max(0.0, 1.0 - quadity - 0.05));

            // 3. Edge Straightness Error
            double sum_edge_error = 0;
            for (const Pt& p : hull) {
                double min_d = 1e18;
                for (int i = 0; i < 4; ++i) {
                    min_d = min(min_d, distToSegment(p, quad[i], quad[(i+1)%4]));
                }
                sum_edge_error += min_d;
            }
            double avg_edge_error = sum_edge_error / hull.size();
            double edge_penalty = exp(-max(0.0, avg_edge_error - 0.5)); // Ignore error < 0.5px

            // 4. Aspect Ratio (Penalize stretched rectangles)
            double min_L = 1e18, max_L = 0;
            for (int i = 0; i < 4; ++i) {
                double dx = quad[i].x - quad[(i+1)%4].x;
                double dy = quad[i].y - quad[(i+1)%4].y;
                double len = sqrt(dx*dx + dy*dy);
                min_L = min(min_L, len);
                max_L = max(max_L, len);
            }
            double aspect = (max_L > 1e-9) ? (min_L / max_L) : 0.0;

            // 5. Final Score Compilation
            double avg_intensity = (double)sum_intensity / count;

            // We want bright, straight-edged, square-proportioned, solid objects
            double score = avg_intensity * fill_penalty * quadity_penalty * edge_penalty * aspect * log10(max(10.0, quad_area));

            if (score > best_score) {
                best_score = score;
                for (int c = 0; c < 4; ++c) best_corners[c] = quad[c];
            }
        }
    }

    for (int c = 0; c < 4; ++c) {
        long long row = llround(best_corners[c].y);
        long long col = llround(best_corners[c].x);
        row = min<long long>(max<long long>(row, 0), M - 1);
        col = min<long long>(max<long long>(col, 0), N - 1);
        cout << row << ' ' << col << '\n';
    }

    return 0;
}