// run_all_tests.cpp  -  Day 1 LeetCode test harness
// -----------------------------------------------------------------------------
// Runs each of the four compiled solutions against every sample and cluster
// test case, verifies the output with a problem-specific checker, prints a
// summary, and writes a full report to  test_results.txt  (overwritten every
// run so the file always reflects the latest results -- keep it in git).
//
// Expected layout (relative to the base directory, see below):
//     solutions-ai-postcomp/Find_the_Square[.exe]
//     solutions-ai-postcomp/Gaussian_Blur[.exe]
//     solutions-ai-postcomp/Largest_Triangle[.exe]
//     solutions-ai-postcomp/Undistortion[.exe]
//     testcases/<problem>/<cluster>/testcase_NN.in   (+ matching .out)
//
// Base directory = argv[1] if given, otherwise the current directory.
// Run it from the  day1-leetcode  folder, or pass that folder as argv[1].
//
// Checkers:
//   * Gaussian Blur, Largest Triangle : exact token-by-token match.
//   * Undistortion                    : every number within 1e-3.
//   * Find the Square                 : 4 corners, any order, +/-1 px per axis.
//
// Requires a C++17 compiler (uses <filesystem>).
// -----------------------------------------------------------------------------
#include <bits/stdc++.h>
#include <filesystem>
namespace fs = std::filesystem;
using namespace std;

enum CheckKind { EXACT, APPROX, SQUARE };

struct Problem {
    string title;     // human-readable
    string dir;       // testcases sub-directory
    string exeBase;   // executable base name (without extension)
    CheckKind kind;
};

// --- Helpers -----------------------------------------------------------------

static string readFile(const fs::path& p) {
    ifstream f(p, ios::binary);
    stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static vector<string> tokens(const string& s) {
    vector<string> out;
    istringstream is(s);
    string t;
    while (is >> t) out.push_back(t);
    return out;
}

// Exact match: identical sequence of whitespace-separated tokens.
static bool checkExact(const string& got, const string& exp) {
    return tokens(got) == tokens(exp);
}

// Approximate match: same count of numbers, each within tolerance.
static bool checkApprox(const string& got, const string& exp) {
    vector<string> g = tokens(got), e = tokens(exp);
    if (g.size() != e.size()) return false;
    for (size_t i = 0; i < g.size(); ++i) {
        try {
            double a = stod(g[i]), b = stod(e[i]);
            if (fabs(a - b) > 1e-3 + 1e-6) return false;
        } catch (...) { return false; }
    }
    return true;
}

// Find-the-Square match: 4 (row,col) corners, any order, +/-1 px on each axis.
static bool checkSquare(const string& got, const string& exp) {
    vector<string> g = tokens(got), e = tokens(exp);
    if (g.size() < 8 || e.size() < 8) return false;

    long long gr[4], gc[4], er[4], ec[4];
    try {
        for (int i = 0; i < 4; ++i) {
            gr[i] = stoll(g[2 * i]); gc[i] = stoll(g[2 * i + 1]);
            er[i] = stoll(e[2 * i]); ec[i] = stoll(e[2 * i + 1]);
        }
    } catch (...) { return false; }

    int perm[4] = {0, 1, 2, 3};
    do {
        bool ok = true;
        for (int i = 0; i < 4 && ok; ++i) {
            int j = perm[i];
            if (llabs(gr[i] - er[j]) > 1 || llabs(gc[i] - ec[j]) > 1) ok = false;
        }
        if (ok) return true;
    } while (next_permutation(perm, perm + 4));
    return false;
}

static bool runCheck(CheckKind k, const string& got, const string& exp) {
    switch (k) {
        case EXACT:  return checkExact(got, exp);
        case APPROX: return checkApprox(got, exp);
        case SQUARE: return checkSquare(got, exp);
    }
    return false;
}

// --- Main --------------------------------------------------------------------

int main(int argc, char** argv) {
    fs::path base = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    base = fs::absolute(base);

    vector<Problem> problems = {
        {"Find the Square",  "find_the_square", "Find_the_Square",  SQUARE},
        {"Gaussian Blur",    "gaussian_blur",   "Gaussian_Blur",    EXACT },
        {"Largest Triangle", "largest_triangle","Largest_Triangle", EXACT },
        {"Undistortion",     "undistortion",    "Undistortion",     APPROX},
    };

    fs::path tmpOut = base / "_runner_tmp_output.txt";

    ostringstream report;
    auto emit = [&](const string& line) {
        cout << line << "\n";
        report << line << "\n";
    };

    {
        time_t now = time(nullptr);
        char ts[64];
        strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", localtime(&now));
        emit("================================================================");
        emit(" Day 1 LeetCode  -  Solution Test Report");
        emit(string(" Generated: ") + ts);
        emit("================================================================");
        emit("");
    }

    int grandPass = 0, grandTotal = 0;
    vector<string> failures;

    for (const Problem& prob : problems) {
        // Locate the compiled solution (with or without a .exe extension).
        fs::path exe = base / "solutions-ai-postcomp" / prob.exeBase;
        fs::path exeExe = exe; exeExe += ".exe";
        fs::path solution;
        if      (fs::exists(exeExe)) solution = exeExe;
        else if (fs::exists(exe))    solution = exe;

        emit("### " + prob.title);

        if (solution.empty()) {
            emit("  SKIPPED - executable not found:");
            emit("    " + exe.string() + "[.exe]");
            emit("    (compile the solution first - see the build instructions)");
            emit("");
            continue;
        }
        emit("  executable: " + solution.string());

        fs::path testRoot = base / "testcases" / prob.dir;
        if (!fs::exists(testRoot)) {
            emit("  SKIPPED - test directory not found: " + testRoot.string());
            emit("");
            continue;
        }

        // Collect every .in file, grouped by its cluster (parent folder name).
        map<string, vector<fs::path>> byCluster;
        for (auto& entry : fs::recursive_directory_iterator(testRoot)) {
            if (entry.is_regular_file() && entry.path().extension() == ".in")
                byCluster[entry.path().parent_path().filename().string()]
                    .push_back(entry.path());
        }
        for (auto& kv : byCluster) sort(kv.second.begin(), kv.second.end());

        // Order clusters with the sample(s) first.
        vector<string> clusterNames;
        for (auto& kv : byCluster) clusterNames.push_back(kv.first);
        sort(clusterNames.begin(), clusterNames.end(),
             [](const string& a, const string& b) {
                 bool sa = a.rfind("sample", 0) == 0;
                 bool sb = b.rfind("sample", 0) == 0;
                 if (sa != sb) return sa;          // samples first
                 return a < b;
             });

        int probPass = 0, probTotal = 0;
        auto t0 = chrono::steady_clock::now();

        for (const string& cname : clusterNames) {
            int cPass = 0, cTotal = 0;
            for (const fs::path& inPath : byCluster[cname]) {
                fs::path expPath = inPath; expPath.replace_extension(".out");
                if (!fs::exists(expPath)) continue;   // no reference output

                string cmd = solution.string() + " < " + inPath.string() +
                             " > " + tmpOut.string();
                int rc = system(cmd.c_str());

                string got = readFile(tmpOut);
                string exp = readFile(expPath);
                bool ok = (rc == 0) && runCheck(prob.kind, got, exp);

                ++cTotal; ++probTotal; ++grandTotal;
                if (ok) { ++cPass; ++probPass; ++grandPass; }
                else    {
                    failures.push_back("  " + prob.title + " / " + cname +
                                       " / " + inPath.stem().string() +
                                       (rc != 0 ? "  (non-zero exit)" : ""));
                }
            }
            if (cTotal > 0) {
                ostringstream l;
                l << "  " << left << setw(16) << cname << right
                  << ": " << setw(4) << cPass << " /" << setw(4) << cTotal
                  << (cPass == cTotal ? "   PASS" : "   FAIL");
                emit(l.str());
            }
        }

        auto t1 = chrono::steady_clock::now();
        double secs = chrono::duration<double>(t1 - t0).count();

        emit("  ------------------------------------------------------------");
        {
            ostringstream l;
            l << "  Problem total: " << probPass << "/" << probTotal
              << " correct";
            if (probTotal > 0)
                l << "  (" << fixed << setprecision(1)
                  << (100.0 * probPass / probTotal) << "%)";
            l << "   [wall time " << fixed << setprecision(2) << secs << " s]";
            emit(l.str());
        }
        emit("");
    }

    emit("================================================================");
    {
        ostringstream l;
        l << " GRAND TOTAL: " << grandPass << "/" << grandTotal
          << " outputs correct";
        if (grandTotal > 0)
            l << "  (" << fixed << setprecision(1)
              << (100.0 * grandPass / grandTotal) << "%)";
        emit(l.str());
    }
    emit(" Incorrect  : " + to_string(grandTotal - grandPass));
    emit("================================================================");

    if (!failures.empty()) {
        emit("");
        emit("Incorrect outputs (problem / cluster / testcase):");
        for (const string& f : failures) emit(f);
    }

    emit("");
    emit("Note: 'Find the Square' is a heuristic vision task with no exact");
    emit("algorithm - partial credit is expected and normal there. The other");
    emit("three problems are deterministic and should reach 100%.");

    // Write the report (truncates any previous version).
    fs::path reportPath = base / "test_results.txt";
    {
        ofstream rf(reportPath, ios::trunc);
        rf << report.str();
    }
    cout << "\nReport written to: " << reportPath.string() << "\n";

    error_code ec;
    fs::remove(tmpOut, ec);   // best-effort cleanup
    return 0;
}
