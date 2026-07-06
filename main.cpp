#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>
#include <pthread.h>

using namespace std;

// --- Wall-clock budget and SIGTERM-safe best-so-far output ----------------
//
// The PACE 2026 heuristic track sends SIGTERM at the 5-minute mark and then
// SIGKILLs the process shortly after. To avoid scoring 0 on instances where
// we are still improving (or still computing the very first solution), we
// keep a best-known forest cached as a single flat string and dump it from
// an async-signal-safe handler.
//
// The cache is double-buffered: solver threads (we only have one, but the
// pattern is still useful) atomically swap in a freshly-built string; the
// signal handler reads the current pointer, writes it to stdout via write(2),
// and _exits. We never call malloc/free in the handler.

namespace deadline {
    using clock_t_ = std::chrono::steady_clock;
    static clock_t_::time_point start_tp;
    static clock_t_::time_point deadline_tp;
    static double budget_seconds_ = 0.0;
    static std::atomic<bool> stop_flag{false};

    static inline void init(double seconds_budget) {
        budget_seconds_ = seconds_budget;
        start_tp = clock_t_::now();
        deadline_tp = start_tp + std::chrono::milliseconds((long long)(seconds_budget * 1000.0));
    }
    static inline double budget_seconds() {
        return budget_seconds_;
    }
    static inline void set_budget_seconds(double seconds_budget) {
        budget_seconds_ = seconds_budget;
    }
    static inline double seconds_left() {
        auto now = clock_t_::now();
        if (now >= deadline_tp) return 0.0;
        return std::chrono::duration<double>(deadline_tp - now).count();
    }
    static inline bool expired() {
        return stop_flag.load(std::memory_order_relaxed) ||
               clock_t_::now() >= deadline_tp;
    }
    static inline clock_t_::time_point current_deadline() {
        return deadline_tp;
    }
    static inline void shorten_for(double seconds) {
        auto temp = clock_t_::now() + std::chrono::milliseconds((long long)(seconds * 1000.0));
        if (temp < deadline_tp) deadline_tp = temp;
    }
    static inline void restore_deadline(clock_t_::time_point saved) {
        deadline_tp = saved;
    }
}

namespace best_cache {
    // Owned strings, indexed by g_slot. The signal handler reads the slot
    // index, then the pointer it stores. Slots are never freed once written.
    static std::atomic<int> g_slot{-1};
    static std::string g_buf[2];
    static std::atomic<const char*> g_ptr[2];
    static std::atomic<size_t> g_len[2];

    // Build a single newline-terminated string of the form
    //   "<line1>\n<line2>\n..."
    // from the current best forest and atomically install it. The solver is
    // single-threaded (only the SIGTERM handler runs asynchronously, and it
    // merely reads the atomics), so no lock is needed here. Avoiding std::mutex
    // also removes any pthread link dependency for portability on the judge.
    static inline void install(const std::vector<std::string>& lines) {
        int next = (g_slot.load(std::memory_order_relaxed) + 1) & 1;
        std::string& buf = g_buf[next];
        buf.clear();
        size_t total = 0;
        for (const auto& s : lines) total += s.size() + 1;
        buf.reserve(total);
        for (const auto& s : lines) { buf.append(s); buf.push_back('\n'); }
        g_ptr[next].store(buf.data(), std::memory_order_release);
        g_len[next].store(buf.size(), std::memory_order_release);
        g_slot.store(next, std::memory_order_release);
    }
    // Async-signal-safe: no locks, no malloc. Read the slot index, then
    // write the string it points to using ::write(2).
    static inline void flush_to_stdout_safely() {
        int slot = g_slot.load(std::memory_order_acquire);
        if (slot < 0) return;
        const char* p = g_ptr[slot].load(std::memory_order_acquire);
        size_t left = g_len[slot].load(std::memory_order_acquire);
        if (p == nullptr || left == 0) return;
        while (left > 0) {
            ssize_t w = ::write(1, p, left);
            if (w <= 0) break;
            p += w;
            left -= (size_t)w;
        }
    }
}

extern "C" void pace_signal_handler(int /*signo*/) {
    // 1) flag the solver so cooperative checks notice
    deadline::stop_flag.store(true, std::memory_order_relaxed);
    // 2) flush whatever best-so-far we have and _exit immediately
    best_cache::flush_to_stdout_safely();
    _exit(0);
}

static double configured_time_budget_seconds() {
    const char* env = std::getenv("PACE_TIME_LIMIT");
    if (env == nullptr || *env == '\0') return 4 * 60 + 58;
    char* end = nullptr;
    double v = std::strtod(env, &end);
    if (end == env || v <= 0.0) return 4 * 60 + 58;
    return std::max(0.1, v);
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t rotl64(uint64_t x, int r) {
    r &= 63;
    return r ? ((x << r) | (x >> (64 - r))) : x;
}

struct NodeKey {
    int sz = 0;
    uint64_t sum1 = 0, sum2 = 0, xor1 = 0, xor2 = 0, top1 = 0, top2 = 0;
    bool operator==(const NodeKey& o) const noexcept {
        return sz == o.sz && sum1 == o.sum1 && sum2 == o.sum2 && xor1 == o.xor1 &&
               xor2 == o.xor2 && top1 == o.top1 && top2 == o.top2;
    }
};

struct NodeKeyHash {
    size_t operator()(const NodeKey& k) const noexcept {
        uint64_t h = splitmix64((uint64_t)k.sz);
        h ^= splitmix64(k.sum1 + 0x100000001b3ULL);
        h ^= rotl64(splitmix64(k.sum2 + 0xc6a4a7935bd1e995ULL), 7);
        h ^= rotl64(splitmix64(k.xor1 + 0x9ddfea08eb382d69ULL), 19);
        h ^= rotl64(splitmix64(k.xor2 + 0xd6e8feb86659fd93ULL), 31);
        h ^= rotl64(splitmix64(k.top1 + 0x94d049bb133111ebULL), 43);
        h ^= rotl64(splitmix64(k.top2 + 0xbf58476d1ce4e5b9ULL), 53);
        return (size_t)h;
    }
};

struct LeafSetKey {
    int sz = 0;
    uint64_t sum1 = 0, sum2 = 0, xor1 = 0, xor2 = 0;
    bool operator==(const LeafSetKey& o) const noexcept {
        return sz == o.sz && sum1 == o.sum1 && sum2 == o.sum2 &&
               xor1 == o.xor1 && xor2 == o.xor2;
    }
};

struct LeafSetKeyHash {
    size_t operator()(const LeafSetKey& k) const noexcept {
        uint64_t h = splitmix64((uint64_t)k.sz);
        h ^= splitmix64(k.sum1 + 0xd1b54a32d192ed03ULL);
        h ^= rotl64(splitmix64(k.sum2 + 0x94d049bb133111ebULL), 11);
        h ^= rotl64(splitmix64(k.xor1 + 0xbf58476d1ce4e5b9ULL), 29);
        h ^= rotl64(splitmix64(k.xor2 + 0x9e3779b97f4a7c15ULL), 47);
        return (size_t)h;
    }
};

static inline LeafSetKey leaf_set_key(const NodeKey& k) {
    return LeafSetKey{k.sz, k.sum1, k.sum2, k.xor1, k.xor2};
}

struct VecHash {
    size_t operator()(const vector<int>& v) const noexcept {
        uint64_t h = splitmix64(v.size());
        for (int x : v) h ^= rotl64(splitmix64((uint64_t)x + 0x9e3779b97f4a7c15ULL), x & 63);
        return (size_t)h;
    }
};

static string trim(const string& s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace((unsigned char)s[a])) ++a;
    while (b > a && isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

struct Tree {
    int n = 0;
    int root = -1;
    vector<array<int, 2>> child;
    vector<int> parent;
    vector<int> leaf_label;          // -1 for internal, 0-based leaf for leaves
    vector<int> label_to_node;
    vector<int> post;
    vector<int> depth;
    vector<int> tin, tout;
    vector<vector<int>> up;
    vector<int> subtree_size;
    vector<NodeKey> key;
    vector<int> leaf_order;

    const vector<uint64_t>* leaf_h1 = nullptr;
    const vector<uint64_t>* leaf_h2 = nullptr;

    int new_leaf(int label) {
        int id = (int)child.size();
        child.push_back({-1, -1});
        parent.push_back(-1);
        leaf_label.push_back(label);
        if (label < 0 || label >= n) throw runtime_error("leaf label out of range");
        if (label_to_node[label] != -1) throw runtime_error("duplicate leaf label");
        label_to_node[label] = id;
        return id;
    }

    int new_internal(int left, int right) {
        int id = (int)child.size();
        child.push_back({left, right});
        parent.push_back(-1);
        leaf_label.push_back(-1);
        parent[left] = id;
        parent[right] = id;
        return id;
    }

    void parse_newick(int n_, const string& raw, const vector<uint64_t>& h1, const vector<uint64_t>& h2) {
        n = n_;
        leaf_h1 = &h1;
        leaf_h2 = &h2;
        child.clear(); parent.clear(); leaf_label.clear();
        label_to_node.assign(n, -1);
        vector<int> st;
        string s = trim(raw);
        for (size_t i = 0; i < s.size();) {
            char c = s[i];
            if (c == ';') { ++i; continue; }
            if (c == '(') { st.push_back(-1); ++i; continue; }
            if (c == ',') { ++i; continue; }
            if (c == ')') {
                if (st.size() < 3) throw runtime_error("malformed Newick: short stack at ')'");
                int right = st.back(); st.pop_back();
                int left = st.back(); st.pop_back();
                int marker = st.back(); st.pop_back();
                if (marker != -1 || left < 0 || right < 0) throw runtime_error("malformed Newick: expected binary pair");
                int id = new_internal(left, right);
                st.push_back(id);
                ++i;
                continue;
            }
            if (isdigit((unsigned char)c)) {
                int val = 0;
                while (i < s.size() && isdigit((unsigned char)s[i])) {
                    val = val * 10 + (s[i] - '0');
                    ++i;
                }
                int label = val - 1;
                st.push_back(new_leaf(label));
                continue;
            }
            if (isspace((unsigned char)c)) { ++i; continue; }
            throw runtime_error(string("unexpected Newick character: ") + c);
        }
        if (st.size() != 1 || st[0] < 0) throw runtime_error("malformed Newick: expression did not reduce to one root");
        root = st[0];
        for (int label = 0; label < n; ++label) {
            if (label_to_node[label] < 0) throw runtime_error("missing leaf label in tree");
        }
        compute_metadata();
    }

    void compute_metadata() {
        int m = (int)child.size();
        post.clear(); post.reserve(m);
        vector<int> order; order.reserve(m);
        vector<int> st = {root};
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            order.push_back(u);
            if (child[u][0] != -1) st.push_back(child[u][0]);
            if (child[u][1] != -1) st.push_back(child[u][1]);
        }
        post = order;
        reverse(post.begin(), post.end());

        subtree_size.assign(m, 0);
        key.assign(m, {});
        for (int u : post) {
            if (child[u][0] == -1) {
                int lab = leaf_label[u];
                subtree_size[u] = 1;
                key[u].sz = 1;
                key[u].sum1 = (*leaf_h1)[lab];
                key[u].sum2 = (*leaf_h2)[lab];
                key[u].xor1 = (*leaf_h1)[lab];
                key[u].xor2 = (*leaf_h2)[lab];
                key[u].top1 = splitmix64((uint64_t)lab + 0x123456789abcdefULL);
                key[u].top2 = splitmix64((uint64_t)lab + 0xfedcba987654321ULL);
            } else {
                int a = child[u][0], b = child[u][1];
                subtree_size[u] = subtree_size[a] + subtree_size[b];
                key[u].sz = subtree_size[u];
                key[u].sum1 = key[a].sum1 + key[b].sum1;
                key[u].sum2 = key[a].sum2 + key[b].sum2;
                key[u].xor1 = key[a].xor1 ^ key[b].xor1;
                key[u].xor2 = key[a].xor2 ^ key[b].xor2;
                pair<uint64_t,uint64_t> A{key[a].top1, key[a].top2};
                pair<uint64_t,uint64_t> B{key[b].top1, key[b].top2};
                if (B < A) swap(A, B);
                key[u].top1 = splitmix64(A.first ^ rotl64(B.first, 17) ^ 0xa24baed4963ee407ULL);
                key[u].top2 = splitmix64(A.second ^ rotl64(B.second, 29) ^ 0x9fb21c651e98df25ULL);
            }
        }

        depth.assign(m, 0);
        tin.assign(m, 0);
        tout.assign(m, 0);
        int K = 1;
        while ((1 << K) <= max(1, m)) ++K;
        up.assign(K, vector<int>(m, root));
        vector<int> stack2 = {root};
        parent[root] = root;
        up[0][root] = root;
        while (!stack2.empty()) {
            int u = stack2.back(); stack2.pop_back();
            for (int v : child[u]) if (v != -1) {
                depth[v] = depth[u] + 1;
                up[0][v] = u;
                stack2.push_back(v);
            }
        }
        int timer = 0;
        vector<pair<int,int>> dfs_stack;
        dfs_stack.reserve(m * 2);
        dfs_stack.push_back({root, 0});
        while (!dfs_stack.empty()) {
            auto [u, state] = dfs_stack.back();
            dfs_stack.pop_back();
            if (state == 0) {
                tin[u] = timer++;
                dfs_stack.push_back({u, 1});
                if (child[u][1] != -1) dfs_stack.push_back({child[u][1], 0});
                if (child[u][0] != -1) dfs_stack.push_back({child[u][0], 0});
            } else {
                tout[u] = timer;
            }
        }
        for (int k = 1; k < K; ++k) {
            for (int u = 0; u < m; ++u) up[k][u] = up[k - 1][up[k - 1][u]];
        }

        leaf_order.clear(); leaf_order.reserve(n);
        vector<int> stack3 = {root};
        while (!stack3.empty()) {
            int u = stack3.back(); stack3.pop_back();
            if (child[u][0] == -1) {
                leaf_order.push_back(leaf_label[u]);
            } else {
                // left-to-right order as represented by the input.
                stack3.push_back(child[u][1]);
                stack3.push_back(child[u][0]);
            }
        }
    }

    int lca(int a, int b) const {
        if (depth[a] < depth[b]) swap(a, b);
        int diff = depth[a] - depth[b];
        for (int k = 0; diff; ++k, diff >>= 1) if (diff & 1) a = up[k][a];
        if (a == b) return a;
        for (int k = (int)up.size() - 1; k >= 0; --k) {
            if (up[k][a] != up[k][b]) {
                a = up[k][a];
                b = up[k][b];
            }
        }
        return up[0][a];
    }

    int lca_labels(const vector<int>& labels) const {
        int x = label_to_node[labels[0]];
        for (size_t i = 1; i < labels.size(); ++i) x = lca(x, label_to_node[labels[i]]);
        return x;
    }

    bool is_ancestor_node(int anc, int desc) const {
        return tin[anc] <= tin[desc] && tin[desc] < tout[anc];
    }

    int child_below(int anc, int desc) const {
        if (anc == desc) return desc;
        int target_depth = depth[anc] + 1;
        int u = desc;
        int diff = depth[u] - target_depth;
        for (int k = 0; diff; ++k, diff >>= 1) if (diff & 1) u = up[k][u];
        return u;
    }

    NodeKey key_for_labels(const vector<int>& labels) const {
        NodeKey k;
        k.sz = (int)labels.size();
        for (int lab : labels) {
            k.sum1 += (*leaf_h1)[lab];
            k.sum2 += (*leaf_h2)[lab];
            k.xor1 ^= (*leaf_h1)[lab];
            k.xor2 ^= (*leaf_h2)[lab];
        }
        return k;
    }

    bool labels_form_clade(const vector<int>& labels, int& node_out) const {
        if (labels.empty()) return false;
        int node = lca_labels(labels);
        if (subtree_size[node] != (int)labels.size()) return false;
        NodeKey lk = key_for_labels(labels);
        const NodeKey& nk = key[node];
        if (lk.sum1 != nk.sum1 || lk.sum2 != nk.sum2 || lk.xor1 != nk.xor1 || lk.xor2 != nk.xor2) return false;
        node_out = node;
        return true;
    }

    vector<int> subtree_labels(int node) const {
        vector<int> labels;
        labels.reserve(subtree_size[node]);
        vector<int> st = {node};
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            if (child[u][0] == -1) labels.push_back(leaf_label[u]);
            else {
                st.push_back(child[u][0]);
                st.push_back(child[u][1]);
            }
        }
        sort(labels.begin(), labels.end());
        return labels;
    }

    vector<int> subtree_edges(int node) const {
        vector<int> edges;
        edges.reserve(max(0, 2 * subtree_size[node] - 2));
        vector<int> st = {node};
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            for (int v : child[u]) if (v != -1) {
                edges.push_back(v);  // edge parent[v] -> v
                st.push_back(v);
            }
        }
        return edges;
    }

    string subtree_newick(int node) const {
        string out;
        out.reserve((size_t)subtree_size[node] * 8);
        vector<pair<int,int>> st;
        st.push_back({node, 0});
        while (!st.empty()) {
            auto [u, state] = st.back(); st.pop_back();
            if (child[u][0] == -1) {
                out += to_string(leaf_label[u] + 1);
                continue;
            }
            if (state == 0) {
                out.push_back('(');
                st.push_back({u, 1});
                st.push_back({child[u][0], 0});
            } else if (state == 1) {
                out.push_back(',');
                st.push_back({u, 2});
                st.push_back({child[u][1], 0});
            } else {
                out.push_back(')');
            }
        }
        return out;
    }

    string subtree_newick_remap(int node, const vector<int>& label_to_new) const {
        string out;
        out.reserve((size_t)subtree_size[node] * 8);
        vector<pair<int,int>> st;
        st.push_back({node, 0});
        while (!st.empty()) {
            auto [u, state] = st.back(); st.pop_back();
            if (child[u][0] == -1) {
                int mapped = label_to_new[leaf_label[u]];
                if (mapped < 0) throw runtime_error("subtree remap missing leaf");
                out += to_string(mapped + 1);
                continue;
            }
            if (state == 0) {
                out.push_back('(');
                st.push_back({u, 1});
                st.push_back({child[u][0], 0});
            } else if (state == 1) {
                out.push_back(',');
                st.push_back({u, 2});
                st.push_back({child[u][1], 0});
            } else {
                out.push_back(')');
            }
        }
        return out;
    }

    string restricted_newick_rec(const vector<int>& labels) const {
        if (labels.size() == 1) return to_string(labels[0] + 1);
        int node = lca_labels(labels);
        vector<int> a, b;
        int first_child = -1;
        for (int lab : labels) {
            int c = child_below(node, label_to_node[lab]);
            if (first_child == -1) first_child = c;
            if (c == first_child) a.push_back(lab); else b.push_back(lab);
        }
        if (b.empty()) return restricted_newick_rec(a); // suppressed degree-2 path
        sort(a.begin(), a.end());
        sort(b.begin(), b.end());
        string sa = restricted_newick_rec(a);
        string sb = restricted_newick_rec(b);
        if (sb < sa) swap(sa, sb);
        return string("(") + sa + "," + sb + ")";
    }

    string restricted_newick(vector<int> labels) const {
        sort(labels.begin(), labels.end());
        return restricted_newick_rec(labels);
    }

    // Canonical 128-bit hash of the restricted rooted topology T|labels,
    // computed without materializing a Newick string. Two label sets yield
    // equal hashes iff their restricted_newick strings are equal; with 128 bits
    // a collision between non-isomorphic topologies is effectively impossible,
    // so this safely replaces the string-equality agreement test, which
    // dominates runtime on large instances.
    pair<uint64_t,uint64_t> restricted_topology_hash_rec(const vector<int>& labels) const {
        if (labels.size() == 1) {
            uint64_t x = (uint64_t)labels[0];
            return { splitmix64(x + 0x123456789abcdefULL), splitmix64(x + 0xfedcba987654321ULL) };
        }
        int node = lca_labels(labels);
        vector<int> a, b;
        int first_child = -1;
        for (int lab : labels) {
            int c = child_below(node, label_to_node[lab]);
            if (first_child == -1) first_child = c;
            if (c == first_child) a.push_back(lab); else b.push_back(lab);
        }
        if (b.empty()) return restricted_topology_hash_rec(a); // suppressed degree-2 path
        auto ha = restricted_topology_hash_rec(a);
        auto hb = restricted_topology_hash_rec(b);
        if (hb < ha) std::swap(ha, hb);
        return { splitmix64(ha.first ^ rotl64(hb.first, 23) ^ 0x9e3779b97f4a7c15ULL),
                 splitmix64(ha.second ^ rotl64(hb.second, 29) ^ 0xc2b2ae3d27d4eb4fULL) };
    }
    // O(size log size) replacement for restricted_topology_hash_rec.
    // Builds the homeomorphic restriction T|labels as a "virtual tree" over the
    // leaves and their pairwise (consecutive-in-tin) LCAs, then folds the SAME
    // canonical combine rule bottom-up. For a binary input every retained
    // internal node has exactly two children, so the produced hash is identical
    // to the recursion's, byte for byte (checked under PACE_HASH_VERIFY).
    pair<uint64_t,uint64_t> restricted_topology_hash_fast(const vector<int>& labels) const {
        int sz = (int)labels.size();
        if (sz == 1) {
            uint64_t x = (uint64_t)labels[0];
            return { splitmix64(x + 0x123456789abcdefULL), splitmix64(x + 0xfedcba987654321ULL) };
        }
        // 1) leaf node ids, sorted by Euler in-time
        vector<int> pts;
        pts.reserve((size_t)sz * 2);
        for (int lab : labels) pts.push_back(label_to_node[lab]);
        auto by_tin = [&](int a, int b) { return tin[a] < tin[b]; };
        sort(pts.begin(), pts.end(), by_tin);
        // 2) add LCAs of consecutive leaves; these are the internal nodes
        int base = (int)pts.size();
        for (int i = 0; i + 1 < base; ++i) pts.push_back(lca(pts[i], pts[i + 1]));
        sort(pts.begin(), pts.end(), by_tin);
        pts.erase(unique(pts.begin(), pts.end()), pts.end());
        int m = (int)pts.size();
        // 3) build parent pointers via a monotonic ancestor stack (positions)
        vector<int> par(m, -1);
        vector<int> stk;
        stk.reserve(m);
        for (int i = 0; i < m; ++i) {
            int u = pts[i];
            while (!stk.empty() && !is_ancestor_node(pts[stk.back()], u)) stk.pop_back();
            par[i] = stk.empty() ? -1 : stk.back();
            stk.push_back(i);
        }
        // 4) fold child hashes into parents. pts is sorted by tin, and an
        //    ancestor has strictly smaller tin than its descendants, so walking
        //    positions high->low visits every node before its parent.
        vector<int> have(m, 0);               // children-folded count per node
        vector<pair<uint64_t,uint64_t>> acc(m);
        int root_pos = 0;
        for (int i = m - 1; i >= 0; --i) {
            pair<uint64_t,uint64_t> hi;
            if (have[i] == 0) {
                // virtual leaf: must be an actual input leaf
                int lab = leaf_label[pts[i]];
                hi = { splitmix64((uint64_t)lab + 0x123456789abcdefULL),
                       splitmix64((uint64_t)lab + 0xfedcba987654321ULL) };
            } else {
                hi = acc[i];
            }
            int p = par[i];
            if (p < 0) { root_pos = i; continue; }
            if (have[p] == 0) { acc[p] = hi; have[p] = 1; }
            else {
                // canonical pairwise combine, identical to the recursion
                pair<uint64_t,uint64_t> ha = acc[p], hb = hi;
                if (hb < ha) std::swap(ha, hb);
                acc[p] = { splitmix64(ha.first  ^ rotl64(hb.first, 23)  ^ 0x9e3779b97f4a7c15ULL),
                           splitmix64(ha.second ^ rotl64(hb.second, 29) ^ 0xc2b2ae3d27d4eb4fULL) };
                ++have[p];
            }
        }
        return acc[root_pos];
    }

    pair<uint64_t,uint64_t> restricted_topology_hash(vector<int> labels) const {
        sort(labels.begin(), labels.end());
#ifdef PACE_HASH_VERIFY
        auto slow = restricted_topology_hash_rec(labels);
        auto fast = restricted_topology_hash_fast(labels);
        if (slow != fast) {
            fprintf(stderr, "HASH MISMATCH size=%zu slow=(%llu,%llu) fast=(%llu,%llu)\n",
                    labels.size(),
                    (unsigned long long)slow.first, (unsigned long long)slow.second,
                    (unsigned long long)fast.first, (unsigned long long)fast.second);
            std::abort();
        }
        return fast;
#else
        return restricted_topology_hash_fast(labels);
#endif
    }

    vector<int> restriction_edges_small(const vector<int>& labels, vector<int>& stamp, int& cur_stamp) const {
        vector<int> edges;
        if (labels.size() <= 1) return edges;
        int root_lca = lca_labels(labels);
        ++cur_stamp;
        if (cur_stamp == numeric_limits<int>::max()) {
            fill(stamp.begin(), stamp.end(), 0);
            cur_stamp = 1;
        }
        for (int lab : labels) {
            int u = label_to_node[lab];
            // Break on the first already-stamped node: within one call all walks
            // share root_lca and cur_stamp, so once a node is stamped every
            // ancestor of it up to root_lca is stamped too. Same edge set as the
            // full walk, but O(output) instead of O(size * depth). (was v5(a))
            while (u != root_lca && stamp[u] != cur_stamp) {
                stamp[u] = cur_stamp;
                edges.push_back(u);
                u = parent[u];
            }
        }
        return edges;
    }
};

// Forward decl: kernelization output expander (defined before main()).
namespace kern { inline string expand_component(const string&, const vector<string>&); }

// ============================================================================
// Exact rooted-MAF branch-and-bound (Whidden-Beiko-Zeh), node-capped.  [DEV]
// ----------------------------------------------------------------------------
// Faithful C++ port of research/maf_engine.py:solve_maf (fuzz-verified there
// against the brute-force oracle). Used as a never-regress core solver on small
// clean clusters: if it FINISHES within the node cap it returns the PROVEN
// optimum (a partition of the local labels 0..m-1); if the cap is hit it
// returns nullopt and the caller keeps the heuristic result. F1 is the
// reference (contracted / finalized); branches cut only F2 edges.
// ============================================================================
namespace exactmaf {

struct Forest {
    vector<int> par, L, R;
    int add_leaf() { int i=(int)par.size(); par.push_back(-1); L.push_back(-1); R.push_back(-1); return i; }
    int add_internal(int a,int b){ int i=(int)par.size(); par.push_back(-1); L.push_back(a); R.push_back(b); par[a]=i; par[b]=i; return i; }
    bool is_leaf(int x) const { return L[x]==-1; }
    int root(int x) const { while(par[x]!=-1) x=par[x]; return x; }
    int sibling(int x) const { int p=par[x]; if(p==-1) return -1; return L[p]==x?R[p]:L[p]; }
    void cut_above(int x){
        int p=par[x]; if(p==-1) return;
        int s=(L[p]==x?R[p]:L[p]); int g=par[p];
        par[x]=-1; par[s]=g;
        if(g!=-1){ if(L[g]==p) L[g]=s; else R[g]=s; }
        par[p]=-1; L[p]=-1; R[p]=-1;   // p detached (garbage; never referenced again)
    }
    int contract_cherry(int a,int b){
        int p=par[a]; L[p]=-1; R[p]=-1; par[a]=-1; par[b]=-1; return p;
    }
};

struct State {
    Forest f1, f2;
    vector<int> f1n, f2n;          // label -> node id
    vector<vector<int>> lset;      // label -> local-leaf list
    vector<char> active;           // label -> active?
    vector<int> Rt;                // active labels (kept clean)
    vector<vector<int>> Rd;        // finalized components (leaf lists)
};

inline int parse_nwk(const string& s, size_t& p, vector<int>& l, vector<int>& r, vector<int>& lb){
    while(p<s.size() && (s[p]==' '||s[p]=='\t')) ++p;
    if(s[p]=='('){
        ++p; int a=parse_nwk(s,p,l,r,lb);
        while(p<s.size() && s[p]!=',') ++p; ++p;
        int b=parse_nwk(s,p,l,r,lb);
        while(p<s.size() && s[p]!=')') ++p; ++p;
        int id=(int)l.size(); l.push_back(a); r.push_back(b); lb.push_back(-1); return id;
    }
    size_t j=p; while(p<s.size() && isdigit((unsigned char)s[p])) ++p;
    int v=std::stoi(s.substr(j,p-j))-1;
    int id=(int)l.size(); l.push_back(-1); r.push_back(-1); lb.push_back(v); return id;
}

inline void build(const string& nw, int m, Forest& f, vector<int>& lab2node){
    vector<int> l,r,lb; size_t p=0;
    int root=parse_nwk(nw,p,l,r,lb); (void)root;
    lab2node.assign(m,-1);
    vector<int> mp(l.size());
    for(int v=0; v<(int)l.size(); ++v){
        if(l[v]==-1){
            mp[v]=f.add_leaf();
            // Bounds-guard the label index: an out-of-range label (malformed /
            // non-contiguous newick) must not write past lab2node. Leaving the
            // slot -1 makes solve() bail via its l2n0[i]<0 guard.
            if(lb[v]>=0 && lb[v]<m) lab2node[lb[v]]=mp[v];
        }
        else mp[v]=f.add_internal(mp[l[v]], mp[r[v]]);
    }
}

inline void erase_label(vector<int>& Rt, int x){
    for(size_t i=0;i<Rt.size();++i) if(Rt[i]==x){ Rt[i]=Rt.back(); Rt.pop_back(); return; }
}

struct Pick { bool done; int a, c; };

inline Pick reduce_and_pick(State& st){
    while(true){
        if((int)st.Rt.size()<=2) return {true,-1,-1};
        // Step 3: finalize any active label whose F2 node is a root
        int fin=-1;
        for(int r : st.Rt){ if(st.f2.par[st.f2n[r]]==-1){ fin=r; break; } }
        if(fin!=-1){
            st.Rd.push_back(st.lset[fin]);
            st.f1.cut_above(st.f1n[fin]);
            st.active[fin]=0; erase_label(st.Rt, fin);
            continue;
        }
        // Step 4: sibling pair in F1, both active labels
        unordered_map<int,int> node2lab; node2lab.reserve(st.Rt.size()*2+1);
        for(int r : st.Rt) node2lab[st.f1n[r]]=r;
        int A=-1,C=-1;
        for(int a : st.Rt){
            int na=st.f1n[a]; int p=st.f1.par[na];
            if(p==-1) continue;
            int sib=st.f1.sibling(na);
            if(sib>=0 && st.f1.is_leaf(sib)){
                auto it=node2lab.find(sib);
                if(it!=node2lab.end() && it->second!=a){ A=a; C=it->second; break; }
            }
        }
        if(A==-1) return {true,-1,-1};
        // Step 5: if (A,C) are also siblings in F2 -> grow (contract in both)
        int pa=st.f2.par[st.f2n[A]];
        if(pa!=-1 && pa==st.f2.par[st.f2n[C]]){
            int nl=(int)st.f1n.size();
            int p1=st.f1.contract_cherry(st.f1n[A], st.f1n[C]);
            int p2=st.f2.contract_cherry(st.f2n[A], st.f2n[C]);
            vector<int> merged=st.lset[A];
            merged.insert(merged.end(), st.lset[C].begin(), st.lset[C].end());
            st.f1n.push_back(p1); st.f2n.push_back(p2);
            st.lset.push_back(std::move(merged)); st.active.push_back(1);
            st.active[A]=0; st.active[C]=0;
            erase_label(st.Rt,A); erase_label(st.Rt,C); st.Rt.push_back(nl);
            continue;
        }
        return {false, A, C};
    }
}

inline vector<vector<int>> collect_partition(const State& st){
    vector<vector<int>> parts = st.Rd;
    unordered_map<int, vector<int>> byroot; byroot.reserve(st.Rt.size()*2+1);
    for(int r : st.Rt){
        int rt=st.f2.root(st.f2n[r]);
        auto& v=byroot[rt];
        v.insert(v.end(), st.lset[r].begin(), st.lset[r].end());
    }
    for(auto& kv : byroot) parts.push_back(std::move(kv.second));
    return parts;
}

inline void path_pendants(const Forest& F, int x, int y, vector<int>& pend){
    vector<int> ax; { int u=x; while(u!=-1){ ax.push_back(u); u=F.par[u]; } }
    unordered_map<int,int> axidx; axidx.reserve(ax.size()*2+1);
    for(int i=0;i<(int)ax.size();++i) axidx[ax[i]]=i;
    int u=y; while(axidx.find(u)==axidx.end()) u=F.par[u];
    int idx=axidx[u];
    pend.clear();
    for(int i=0;i<idx;++i){ if(i==idx-1) continue; pend.push_back(F.sibling(ax[i])); }
    vector<int> ay; { int w=y; while(axidx.find(w)==axidx.end()){ ay.push_back(w); w=F.par[w]; } }
    for(int i=0;i<(int)ay.size();++i){ if(i==(int)ay.size()-1) continue; pend.push_back(F.sibling(ay[i])); }
}

// node + wall-time budget guard (per-solve); set in solve()
struct Budget { long long nodes=0, cap=0; double abort_left=-1e18; bool aborted=false; };

// diagnostics (PACE_KDEBUG)
inline long long& dbg_calls(){ static long long v=0; return v; }
inline long long& dbg_done(){ static long long v=0; return v; }
inline long long& dbg_capped(){ static long long v=0; return v; }
inline long long& dbg_maxm_done(){ static long long v=0; return v; }

inline vector<vector<int>> maf_min(State st, int ub, Budget& bg){
    if(bg.aborted) return {};
    if(++bg.nodes > bg.cap){ bg.aborted=true; return {}; }
    // Wall-time safety: bound each exact call so a hard cluster can never eat
    // into the anytime budget. Checked every 8192 nodes (cheap).
    if((bg.nodes & 8191)==0 && deadline::seconds_left() < bg.abort_left){ bg.aborted=true; return {}; }
    Pick pk=reduce_and_pick(st);
    if((int)st.Rd.size() >= ub) return {};
    if(pk.done) return collect_partition(st);
    int a=pk.a, c=pk.c;
    int xa=st.f2n[a], xc=st.f2n[c];
    vector<char> br; vector<int> pend;
    if(st.f2.root(xa)!=st.f2.root(xc)) br={'a','c'};
    else {
        path_pendants(st.f2, xa, xc, pend);
        br = (pend.size()==1) ? vector<char>{'p'} : vector<char>{'p','a','c'};
    }
    vector<vector<int>> best; int bestlen=ub;
    for(char b : br){
        State ns=st;
        if(b=='a') ns.f2.cut_above(ns.f2n[a]);
        else if(b=='c') ns.f2.cut_above(ns.f2n[c]);
        else for(int pp : pend) ns.f2.cut_above(pp);
        auto res=maf_min(std::move(ns), bestlen, bg);
        if(bg.aborted) return {};
        if(!res.empty() && (int)res.size() < bestlen){ best=std::move(res); bestlen=(int)best.size(); }
    }
    return best;
}

// Solve the exact rooted MAF of the m-leaf sub-instance (newicks with 1-based
// local labels). Returns the optimal partition of labels 0..m-1, or nullopt if
// the node cap was hit before completion.
inline optional<vector<vector<int>>> solve(const string& nw0, const string& nw1, int m,
                                           long long node_cap, double max_seconds){
    if(m<=0) return nullopt;
    if(m==1) return vector<vector<int>>{{0}};
    State st;
    vector<int> l2n0, l2n1;
    build(nw0, m, st.f1, l2n0);
    build(nw1, m, st.f2, l2n1);
    st.f1n.resize(m); st.f2n.resize(m); st.lset.resize(m); st.active.assign(m,1); st.Rt.resize(m);
    for(int i=0;i<m;++i){
        if(l2n0[i]<0 || l2n1[i]<0) return nullopt;   // malformed / non-contiguous labels
        st.f1n[i]=l2n0[i]; st.f2n[i]=l2n1[i]; st.lset[i]={i}; st.Rt[i]=i;
    }
    Budget bg; bg.cap=node_cap;
    // abort if we drop below (now - max_seconds) OR below a hard 0.3s floor.
    bg.abort_left = max(deadline::seconds_left() - max_seconds, 0.3);
    ++dbg_calls();
    auto res=maf_min(std::move(st), m+1, bg);
    if(bg.aborted || res.empty()){ ++dbg_capped(); return nullopt; }
    ++dbg_done(); if(m > dbg_maxm_done()) dbg_maxm_done()=m;
    return res;
}

} // namespace exactmaf

struct Component {
    int id = -1;
    bool active = true;
    int size = 0;
    vector<int> labels;
    string newick;
    array<vector<int>, 2> edges;
    array<int, 2> clade_root{{-1, -1}};
    NodeKey leaf_key;
};

struct Solver {
    int n = 0;
    array<Tree, 2> T;
    vector<uint64_t> leaf_h1, leaf_h2;
    vector<Component> comps;
    vector<int> label_to_comp;
    array<vector<unsigned char>, 2> used;
    array<vector<int>, 2> tmp_stamp;
    array<int, 2> stamp_counter{{1, 1}};
    bool publish_cache_ = true;
    // Kernelization: if non-empty, the Solver operates on a reduced (kernel)
    // instance and every emitted component newick is expanded back to original
    // labels via expand_nwk_[kernel_leaf_0based]. Empty => identity (no kernel).
    vector<string> expand_nwk_;
    // When kernelizing, seed the restart RNG from the ORIGINAL n (not the kernel
    // size K) so the search trajectory is comparable to the non-kernel run and
    // the smaller kernel is a pure advantage rather than a trajectory change.
    int orig_n_ = 0;   // 0 => use n
    // Exact-core repair: use the verified Whidden B&B (exactmaf::) to solve small
    // clean clusters to proven optimum inside the cluster-repair path (never
    // regress: only used when it FINISHES and beats the heuristic sub-result).
    bool use_exact_core_ = (std::getenv("PACE_NOEXACT") == nullptr);
    static constexpr int EXACT_CORE_MAX_LEAVES = 128;
    static constexpr long long EXACT_CORE_NODE_CAP = 2000000LL;
    // Clades the exact B&B could not solve within the cap. Their difficulty is a
    // fixed property of the (leaf set, both trees), so re-solving on later passes
    // just wastes budget -> memoize and skip. Skipping only ever costs a missed
    // gain, never validity.
    unordered_map<LeafSetKey, char, LeafSetKeyHash> exact_capped_;

    static constexpr int SMALL_RESTRICT_LIMIT = 44;
    static constexpr int MERGE_SMALL_LIMIT = 72;
    static constexpr int SINGLETON_PAIR_ORDER_RADIUS = 28;
    static constexpr int PAIR_SEED_ORDER_RADIUS = 18;
    static constexpr int PACK_MIN = 3;
    static constexpr int PACK_MAX = 12;
    static constexpr int SINGLETON_PACK_BB_MAX_CANDIDATES = 650;
    static constexpr int SINGLETON_PACK_BB_MAX_NODES = 20000;
    static constexpr int MERGE_PASSES = 8;
    static constexpr int COMPONENT_PACK_MAX_COMPONENTS = 9;
    static constexpr int COMPONENT_PACK_MAX_LEAVES = 72;
    static constexpr int COMPONENT_PACK_MAX_CANDIDATES = 25000;
    static constexpr int COMPONENT_PACK_BB_MAX_CANDIDATES = 650;
    static constexpr int COMPONENT_PACK_BB_MAX_NODES = 20000;
    static constexpr int COMPONENT_REPAIR_MAX_WINDOWS = 20;
    static constexpr int COMPONENT_REPAIR_MAX_CANDIDATES = 16000;
    static constexpr int COMPONENT_REPAIR_MAX_SEARCH_NODES = 45000;
    static constexpr int CORRIDOR_MERGE_MAX_WINDOWS = 8000;
    static constexpr int CORRIDOR_MERGE_MAX_IDS = 24;
    static constexpr int CORRIDOR_MERGE_MAX_LEAVES = 192;
    static constexpr int CORRIDOR_WINDOW_MAX_COMPONENTS = 24;
    static constexpr int CORRIDOR_WINDOW_MAX_LEAVES = 192;
    static constexpr int CORRIDOR_WINDOW_MAX_CANDIDATES = 40000;
    static constexpr int FREE_REGION_PAIR_ALL_LIMIT = 72;
    static constexpr int FREE_REGION_PAIR_RADIUS = 10;
    static constexpr int FREE_REGION_PAIR_HASH_TRIES = 4;
    static constexpr int LOCAL_MAX_LEAVES = 14;
    static constexpr int LOCAL_MAX_COMPONENTS = 12;
    static constexpr int LOCAL_MAX_WINDOWS = 18;
    static constexpr int LOCAL_MAX_CANDIDATES = 5000;
    static constexpr int LOCAL_MAX_SEARCH_NODES = 18000;
    static constexpr int LOCAL_REPAIR_BASE_COMPONENT_LIMIT = 2000;
    static constexpr int LOCAL_REPAIR_COMPONENT_LIMIT = 3000;
    static constexpr int DEEP_LOCAL_MAX_LEAVES = 18;
    static constexpr int DEEP_LOCAL_MAX_COMPONENTS = 14;
    static constexpr int DEEP_LOCAL_MAX_WINDOWS = 3;
    static constexpr int DEEP_LOCAL_MAX_CANDIDATES = 12000;
    static constexpr int DEEP_LOCAL_MAX_SEARCH_NODES = 25000;
    static constexpr int DEEP_LOCAL_MAX_ATTEMPTS = 24;
    // Window leaf cap of 14 (not larger): conflict repair is throughput-bound on
    // large instances, and smaller windows process more conflicts per unit time;
    // a sweep showed 14 ties 16 at the full budget and beats it at shorter ones.
    static constexpr int CONFLICT_REPAIR_MAX_LEAVES = 14;
    static constexpr int CONFLICT_REPAIR_MAX_BLOCKERS = 2;
    static constexpr int CONFLICT_REPAIR_MAX_WINDOWS = 6000;
    static constexpr int CONFLICT_REPAIR_MAX_CANDIDATES = 8000;
    static constexpr int CONFLICT_REPAIR_MAX_SEARCH_NODES = 20000;
    static constexpr int MAST_DP_LEAF_LIMIT = 5500;
    static constexpr int MAST_MIN_GAIN = 8;
    static constexpr int MAST_TOP_CELLS = 4096;
    static constexpr int MAST_MAX_COMPONENT_CANDIDATES = 1200;
    static constexpr int MAST_PACK_BB_MAX_CANDIDATES = 650;
    static constexpr int MAST_PACK_BB_MAX_NODES = 20000;
    static constexpr int EXACT_SMALL_N = 14;
    static constexpr int PORTFOLIO_SEEDS = 8;
    static constexpr int CHERRY_CUT_COMPONENT_LIMIT = 256;
    static constexpr int CHERRY_CUT_MAX_ROUNDS = 90;
    static constexpr int CHERRY_CUT_SIDE_LIMIT = 12;
    static constexpr int CHERRY_CUT_CONFLICT_INSPECT = 2500;
    static constexpr int GREEDY_PARTITION_VERIFY_LIMIT = 192;
    static constexpr int GREEDY_PARTITION_MIN_GAIN = 32;
    static constexpr int GREEDY_PARTITION_VARIANTS = 3;
    static constexpr int COMMON_CLUSTER_MIN_SIZE = 64;
    static constexpr int COMMON_CLUSTER_SCAN_LIMIT = 260;
    static constexpr int SMALL_COMMON_CLUSTER_REPAIR_MAX_ATTEMPTS = 8;

    Solver(int n_, const string& nw0, const string& nw1, bool publish_cache = true)
        : n(n_), publish_cache_(publish_cache) {
        leaf_h1.resize(n);
        leaf_h2.resize(n);
        for (int i = 0; i < n; ++i) {
            leaf_h1[i] = splitmix64((uint64_t)i + 0x243f6a8885a308d3ULL);
            leaf_h2[i] = splitmix64((uint64_t)i + 0x13198a2e03707344ULL);
        }
        T[0].parse_newick(n, nw0, leaf_h1, leaf_h2);
        T[1].parse_newick(n, nw1, leaf_h1, leaf_h2);
        used[0].assign(T[0].child.size(), 0);
        used[1].assign(T[1].child.size(), 0);
        tmp_stamp[0].assign(T[0].child.size(), 0);
        tmp_stamp[1].assign(T[1].child.size(), 0);
        label_to_comp.assign(n, -1);
    }

    NodeKey labels_key(const vector<int>& labels) const {
        NodeKey k;
        k.sz = (int)labels.size();
        for (int lab : labels) {
            k.sum1 += leaf_h1[lab]; k.sum2 += leaf_h2[lab];
            k.xor1 ^= leaf_h1[lab]; k.xor2 ^= leaf_h2[lab];
        }
        return k;
    }

    void fill_leaf_key(Component& c) const {
        c.leaf_key = labels_key(c.labels);
    }

    Component build_singleton(int label) const {
        Component c;
        c.size = 1;
        c.labels = {label};
        c.newick = to_string(label + 1);
        c.clade_root = {T[0].label_to_node[label], T[1].label_to_node[label]};
        c.leaf_key = labels_key(c.labels);
        return c;
    }

    Component build_clade_component(int node0, int node1) const {
        Component c;
        c.size = T[0].subtree_size[node0];
        c.labels = T[0].subtree_labels(node0);
        c.newick = T[0].subtree_newick(node0);
        c.edges[0] = T[0].subtree_edges(node0);
        c.edges[1] = T[1].subtree_edges(node1);
        c.clade_root = {node0, node1};
        c.leaf_key = labels_key(c.labels);
        return c;
    }

    optional<Component> build_component_from_labels(vector<int> labels, int small_limit, bool allow_large_clade) {
        sort(labels.begin(), labels.end());
        labels.erase(unique(labels.begin(), labels.end()), labels.end());
        if (labels.empty()) return nullopt;
        if ((int)labels.size() == 1) return build_singleton(labels[0]);

        if ((int)labels.size() > small_limit) {
            if (!allow_large_clade) return nullopt;
            int n0 = -1, n1 = -1;
            if (!T[0].labels_form_clade(labels, n0)) return nullopt;
            if (!T[1].labels_form_clade(labels, n1)) return nullopt;
            if (!(T[0].key[n0] == T[1].key[n1])) return nullopt;
            return build_clade_component(n0, n1);
        }

        // Agreement test via canonical 128-bit topology hash, with no Newick
        // string built. Candidates are produced in huge numbers but few are
        // accepted, so the (expensive) Newick string is deferred to
        // add_component and built only for components that enter the solution.
        if (T[0].restricted_topology_hash(labels) != T[1].restricted_topology_hash(labels))
            return nullopt;
        Component c;
        c.size = (int)labels.size();
        c.labels = labels;
        c.newick.clear();  // deferred: built lazily in add_component
        c.edges[0] = T[0].restriction_edges_small(labels, tmp_stamp[0], stamp_counter[0]);
        c.edges[1] = T[1].restriction_edges_small(labels, tmp_stamp[1], stamp_counter[1]);
        int r0 = -1, r1 = -1;
        if (T[0].labels_form_clade(labels, r0)) c.clade_root[0] = r0;
        if (T[1].labels_form_clade(labels, r1)) c.clade_root[1] = r1;
        c.leaf_key = labels_key(c.labels);
        return c;
    }

    optional<Component> build_agreement_component_unbounded(vector<int> labels) {
        sort(labels.begin(), labels.end());
        labels.erase(unique(labels.begin(), labels.end()), labels.end());
        if (labels.empty()) return nullopt;
        if ((int)labels.size() == 1) return build_singleton(labels[0]);

        int n0 = -1, n1 = -1;
        bool clade0 = T[0].labels_form_clade(labels, n0);
        bool clade1 = T[1].labels_form_clade(labels, n1);
        if (clade0 && clade1 && T[0].key[n0] == T[1].key[n1]) {
            return build_clade_component(n0, n1);
        }
        if (T[0].restricted_topology_hash(labels) != T[1].restricted_topology_hash(labels))
            return nullopt;

        Component c;
        c.size = (int)labels.size();
        c.labels = labels;
        c.newick.clear();
        c.edges[0] = T[0].restriction_edges_small(labels, tmp_stamp[0], stamp_counter[0]);
        c.edges[1] = T[1].restriction_edges_small(labels, tmp_stamp[1], stamp_counter[1]);
        if (clade0) c.clade_root[0] = n0;
        if (clade1) c.clade_root[1] = n1;
        c.leaf_key = labels_key(c.labels);
        return c;
    }

    bool conflicts_with_used(const Component& cand, const vector<int>& exclude_ids) {
        for (int ti = 0; ti < 2; ++ti) {
            ++stamp_counter[ti];
            if (stamp_counter[ti] == numeric_limits<int>::max()) {
                fill(tmp_stamp[ti].begin(), tmp_stamp[ti].end(), 0);
                stamp_counter[ti] = 1;
            }
            for (int id : exclude_ids) if (id >= 0 && id < (int)comps.size() && comps[id].active) {
                for (int e : comps[id].edges[ti]) tmp_stamp[ti][e] = stamp_counter[ti];
            }
            for (int e : cand.edges[ti]) {
                if (used[ti][e] && tmp_stamp[ti][e] != stamp_counter[ti]) return true;
            }
        }
        return false;
    }

    void mark_edges(const Component& c, bool val) {
        for (int ti = 0; ti < 2; ++ti) {
            for (int e : c.edges[ti]) used[ti][e] = (unsigned char)val;
        }
    }

    int add_component(Component c) {
        c.id = (int)comps.size();
        c.active = true;
        // Build the Newick string here (once) for components whose construction
        // deferred it (build_component_from_labels). Only accepted components
        // reach this point, so rejected candidates never pay for a string.
        if (c.newick.empty() && !c.labels.empty()) c.newick = T[0].restricted_newick(c.labels);
        comps.push_back(std::move(c));
        const Component& ref = comps.back();
        mark_edges(ref, true);
        for (int lab : ref.labels) label_to_comp[lab] = ref.id;
        return ref.id;
    }

    void remove_component(int id) {
        if (id < 0 || id >= (int)comps.size() || !comps[id].active) return;
        mark_edges(comps[id], false);
        comps[id].active = false;
        for (int lab : comps[id].labels) if (label_to_comp[lab] == id) label_to_comp[lab] = -1;
    }

    int active_count() const {
        int c = 0;
        for (auto& x : comps) if (x.active) ++c;
        return c;
    }

    void verify_current_forest(const char* context) const {
#ifdef PACE_LOCAL_VERIFY
        auto fail = [&](const string& msg) -> void {
            cerr << "PACE_LOCAL_VERIFY failed";
            if (context) cerr << " at " << context;
            cerr << ": " << msg << "\n";
            std::abort();
        };

        vector<int> seen_label(n, -1);
        array<vector<unsigned char>, 2> seen_edge;
        seen_edge[0].assign(T[0].child.size(), 0);
        seen_edge[1].assign(T[1].child.size(), 0);

        int active = 0;
        int label_total = 0;
        for (int id = 0; id < (int)comps.size(); ++id) {
            const Component& c = comps[id];
            if (!c.active) continue;
            ++active;
            if (c.id != id) fail("component id mismatch");
            if (c.size != (int)c.labels.size()) fail("component size does not match label vector");
            if (c.labels.empty()) fail("active component has no labels");
            if (c.newick.empty()) fail("active component has empty Newick string");

            vector<int> sorted_labels = c.labels;
            sort(sorted_labels.begin(), sorted_labels.end());
            if (sorted_labels != c.labels) fail("component labels are not sorted");
            if (unique(sorted_labels.begin(), sorted_labels.end()) != sorted_labels.end()) {
                fail("component contains duplicate labels");
            }

            NodeKey lk = labels_key(c.labels);
            if (lk.sz != c.leaf_key.sz || lk.sum1 != c.leaf_key.sum1 ||
                lk.sum2 != c.leaf_key.sum2 || lk.xor1 != c.leaf_key.xor1 ||
                lk.xor2 != c.leaf_key.xor2) {
                fail("component leaf key is stale");
            }

            label_total += (int)c.labels.size();
            for (int lab : c.labels) {
                if (lab < 0 || lab >= n) fail("label out of range");
                if (seen_label[lab] != -1) fail("label appears in multiple components");
                if (label_to_comp[lab] != id) fail("label_to_comp is stale");
                seen_label[lab] = id;
            }

            if (c.labels.size() > 1) {
                string r0 = T[0].restricted_newick(c.labels);
                string r1 = T[1].restricted_newick(c.labels);
                if (r0 != r1) fail("component is not an agreement subtree");
            }

            for (int ti = 0; ti < 2; ++ti) {
                for (int e : c.edges[ti]) {
                    if (e < 0 || e >= (int)seen_edge[ti].size()) fail("edge id out of range");
                    if (seen_edge[ti][e]) fail("edge reused by active components");
                    seen_edge[ti][e] = 1;
                }
            }
        }

        if (active <= 0) fail("no active components");
        if (label_total != n) fail("active components do not partition all labels");
        for (int lab = 0; lab < n; ++lab) {
            if (seen_label[lab] == -1) fail("missing label from active forest");
        }
        for (int ti = 0; ti < 2; ++ti) {
            if (used[ti].size() != seen_edge[ti].size()) fail("used edge vector has wrong size");
            for (int e = 0; e < (int)used[ti].size(); ++e) {
                if ((used[ti][e] != 0) != (seen_edge[ti][e] != 0)) fail("used edge map is stale");
            }
        }
#else
        (void)context;
#endif
    }

    // Always-on output feasibility check (NOT gated by PACE_LOCAL_VERIFY).
    // Recomputes the three judge-relevant properties from the active
    // components' labels: (1) the labels form a partition of 0..n-1,
    // (2) every component is an agreement subtree in BOTH trees, and
    // (3) components are edge-disjoint in both trees. Returns false on any
    // violation so callers can refuse to publish an infeasible forest.
    // Cost O(n log n); used as a submission safety net (one infeasible output
    // disqualifies the whole submission).
    bool current_forest_valid() const {
        vector<unsigned char> seen_label(n, 0);
        array<vector<unsigned char>, 2> seen_edge;
        seen_edge[0].assign(T[0].child.size(), 0);
        seen_edge[1].assign(T[1].child.size(), 0);
        long long total = 0;
        for (const Component& c : comps) {
            if (!c.active) continue;
            if (c.labels.empty()) return false;
            for (int lab : c.labels) {
                if (lab < 0 || lab >= n) return false;
                if (seen_label[lab]) return false;          // overlap / duplicate
                seen_label[lab] = 1; ++total;
            }
            if (c.labels.size() > 1) {
                // Agreement test (component induces the same rooted topology in
                // both trees). Small components use the independent canonical
                // Newick string equality; large ones use the O(size log size)
                // 128-bit topology hash instead, because the string form is
                // O(size^2) in time AND memory on deep (caterpillar) shapes and
                // would OOM/time out on a giant near-identical component (e.g.
                // two identical deep trees -> one size-n component). The hash is
                // the same canonical agreement test used during construction.
                bool agree = (c.size <= 4096)
                    ? (T[0].restricted_newick(c.labels) == T[1].restricted_newick(c.labels))
                    : (T[0].restricted_topology_hash(c.labels) == T[1].restricted_topology_hash(c.labels));
                if (!agree) return false;                    // not an agreement subtree
            }
            for (int ti = 0; ti < 2; ++ti)
                for (int e : c.edges[ti]) {
                    if (e < 0 || e >= (int)seen_edge[ti].size()) return false;
                    if (seen_edge[ti][e]) return false;      // edge reused -> not edge-disjoint
                    seen_edge[ti][e] = 1;
                }
        }
        return total == n;                                  // covers every leaf exactly once
    }

    vector<Component> singleton_seed() const {
        vector<Component> seed;
        seed.reserve(n);
        for (int lab = 0; lab < n; ++lab) seed.push_back(build_singleton(lab));
        return seed;
    }

    vector<Component> common_clade_sweep(int ref_idx) {
        int other_idx = 1 - ref_idx;
        unordered_map<NodeKey, int, NodeKeyHash> other_map;
        other_map.reserve(T[other_idx].child.size() * 2 + 1);
        for (int u = 0; u < (int)T[other_idx].child.size(); ++u) {
            other_map.emplace(T[other_idx].key[u], u);
        }

        vector<int> nodes(T[ref_idx].child.size());
        iota(nodes.begin(), nodes.end(), 0);
        sort(nodes.begin(), nodes.end(), [&](int a, int b) {
            if (T[ref_idx].subtree_size[a] != T[ref_idx].subtree_size[b])
                return T[ref_idx].subtree_size[a] > T[ref_idx].subtree_size[b];
            return a < b;
        });

        vector<unsigned char> blocked(T[ref_idx].child.size(), 0);
        vector<unsigned char> covered(n, 0);
        vector<Component> out;
        out.reserve(n);

        for (int u : nodes) {
            if (deadline::expired()) break;
            int sz = T[ref_idx].subtree_size[u];
            if (sz <= 1) continue;
            if (blocked[u]) continue;
            auto it = other_map.find(T[ref_idx].key[u]);
            if (it == other_map.end()) continue;
            int v = it->second;
            int node0 = (ref_idx == 0 ? u : v);
            int node1 = (ref_idx == 1 ? u : v);
            Component c = build_clade_component(node0, node1);
            // Block the selected reference subtree. Since we process larger clades first,
            // this is enough to maintain laminar disjointness in the reference tree.
            vector<int> st = {u};
            while (!st.empty()) {
                int x = st.back(); st.pop_back();
                blocked[x] = 1;
                if (T[ref_idx].child[x][0] == -1) covered[T[ref_idx].leaf_label[x]] = 1;
                else {
                    st.push_back(T[ref_idx].child[x][0]);
                    st.push_back(T[ref_idx].child[x][1]);
                }
            }
            out.push_back(std::move(c));
        }
        for (int lab = 0; lab < n; ++lab) if (!covered[lab]) out.push_back(build_singleton(lab));
        return out;
    }

    void install_components(vector<Component> initial) {
        comps.clear();
        fill(used[0].begin(), used[0].end(), 0);
        fill(used[1].begin(), used[1].end(), 0);
        fill(label_to_comp.begin(), label_to_comp.end(), -1);
        comps.reserve(initial.size() + n / 2 + 16);
        for (auto& c : initial) add_component(std::move(c));
        verify_current_forest("install_components");
    }

    vector<pair<int,int>> component_pairs_from_tree(int ti) {
        const Tree& tr = T[ti];
        vector<int> first(tr.child.size(), -1), second(tr.child.size(), -1), cnt(tr.child.size(), 0);
        vector<uint64_t> encoded;
        encoded.reserve(comps.size() * 2 + 1);
        auto add_id = [](int id, int& c, int& a, int& b) {
            if (id < 0) return;
            if (c >= 1 && id == a) return;
            if (c >= 2 && id == b) return;
            if (c == 0) { a = id; c = 1; }
            else if (c == 1) { b = id; c = 2; }
            else c = 3;
        };
        for (int u : tr.post) {
            if (tr.child[u][0] == -1) {
                int lab = tr.leaf_label[u];
                int id = label_to_comp[lab];
                if (id >= 0 && comps[id].active) { first[u] = id; cnt[u] = 1; }
            } else {
                int a = -1, b = -1, c = 0;
                for (int ch : tr.child[u]) {
                    if (cnt[ch] == 1) add_id(first[ch], c, a, b);
                    else if (cnt[ch] == 2) { add_id(first[ch], c, a, b); add_id(second[ch], c, a, b); }
                    else if (cnt[ch] > 2) c = 3;
                    if (c > 2) break;
                }
                cnt[u] = min(c, 3);
                first[u] = a; second[u] = b;
                if (c == 2 && a != b && a >= 0 && b >= 0) {
                    if (a > b) swap(a, b);
                    encoded.push_back(((uint64_t)(uint32_t)a << 32) | (uint32_t)b);
                }
            }
        }
        sort(encoded.begin(), encoded.end());
        encoded.erase(unique(encoded.begin(), encoded.end()), encoded.end());
        vector<pair<int,int>> pairs;
        pairs.reserve(encoded.size());
        for (uint64_t x : encoded) pairs.push_back({(int)(x >> 32), (int)(x & 0xffffffffu)});
        return pairs;
    }

    bool try_replace(const vector<int>& old_ids, Component cand) {
        if (conflicts_with_used(cand, old_ids)) return false;
        for (int id : old_ids) remove_component(id);
        add_component(std::move(cand));
        verify_current_forest("try_replace");
        return true;
    }

    int merge_pass() {
        vector<pair<int,int>> pairs = component_pairs_from_tree(0);
        vector<pair<int,int>> p2 = component_pairs_from_tree(1);
        pairs.insert(pairs.end(), p2.begin(), p2.end());
        sort(pairs.begin(), pairs.end());
        pairs.erase(unique(pairs.begin(), pairs.end()), pairs.end());
        sort(pairs.begin(), pairs.end(), [&](auto A, auto B) {
            int sa = comps[A.first].size + comps[A.second].size;
            int sb = comps[B.first].size + comps[B.second].size;
            if (sa != sb) return sa > sb;
            return A < B;
        });
        int merged = 0;
        for (auto [a, b] : pairs) {
            if (deadline::expired()) break;
            if (a < 0 || b < 0 || a >= (int)comps.size() || b >= (int)comps.size()) continue;
            if (!comps[a].active || !comps[b].active) continue;
            vector<int> labels;
            labels.reserve(comps[a].labels.size() + comps[b].labels.size());
            labels.insert(labels.end(), comps[a].labels.begin(), comps[a].labels.end());
            labels.insert(labels.end(), comps[b].labels.begin(), comps[b].labels.end());
            bool allow_large_clade = true;
            auto cand = build_component_from_labels(labels, MERGE_SMALL_LIMIT, allow_large_clade);
            if (!cand) continue;
            if ((int)cand->labels.size() <= max(comps[a].size, comps[b].size)) continue;
            if (try_replace({a, b}, std::move(*cand))) ++merged;
        }
        return merged;
    }

    vector<pair<int,int>> singleton_pair_candidates_from_tree(int ti) {
        const Tree& tr = T[ti];
        vector<int> cnt(tr.child.size(), 0), first(tr.child.size(), -1), second(tr.child.size(), -1);
        vector<pair<int,int>> pairs;
        auto add_label = [](int lab, int& c, int& a, int& b) {
            if (lab < 0) return;
            if (c >= 1 && lab == a) return;
            if (c >= 2 && lab == b) return;
            if (c == 0) { a = lab; c = 1; }
            else if (c == 1) { b = lab; c = 2; }
            else c = 3;
        };
        for (int u : tr.post) {
            if (tr.child[u][0] == -1) {
                int lab = tr.leaf_label[u];
                int cid = label_to_comp[lab];
                if (cid >= 0 && comps[cid].active && comps[cid].size == 1) {
                    cnt[u] = 1;
                    first[u] = lab;
                }
            } else {
                int c = 0, a = -1, b = -1;
                for (int ch : tr.child[u]) {
                    if (cnt[ch] == 1) add_label(first[ch], c, a, b);
                    else if (cnt[ch] == 2) { add_label(first[ch], c, a, b); add_label(second[ch], c, a, b); }
                    else if (cnt[ch] > 2) c = 3;
                    if (c > 2) break;
                }
                cnt[u] = min(c, 3);
                first[u] = a;
                second[u] = b;
                if (c == 2 && a >= 0 && b >= 0) {
                    if (a > b) swap(a, b);
                    pairs.push_back({a, b});
                }
            }
        }
        return pairs;
    }

    int singleton_pair_pass() {
        vector<pair<int,int>> pairs;
        for (int ti = 0; ti < 2; ++ti) {
            auto p = singleton_pair_candidates_from_tree(ti);
            pairs.insert(pairs.end(), p.begin(), p.end());
            vector<int> active_order;
            active_order.reserve(n);
            for (int lab : T[ti].leaf_order) {
                int cid = label_to_comp[lab];
                if (cid >= 0 && comps[cid].active && comps[cid].size == 1) active_order.push_back(lab);
            }
            for (int i = 0; i < (int)active_order.size(); ++i) {
                for (int d = 1; d <= SINGLETON_PAIR_ORDER_RADIUS && i + d < (int)active_order.size(); ++d) {
                    int a = active_order[i], b = active_order[i + d];
                    if (a > b) swap(a, b);
                    pairs.push_back({a, b});
                }
            }
        }

        sort(pairs.begin(), pairs.end());
        pairs.erase(unique(pairs.begin(), pairs.end()), pairs.end());
        struct Scored { int cost; uint64_t key; int a; int b; Component comp; };
        vector<Scored> scored;
        scored.reserve(pairs.size());
        for (auto [a, b] : pairs) {
            int ca = label_to_comp[a], cb = label_to_comp[b];
            if (ca < 0 || cb < 0 || ca == cb) continue;
            if (!comps[ca].active || !comps[cb].active || comps[ca].size != 1 || comps[cb].size != 1) continue;
            vector<int> labels{a, b};
            auto cand = build_component_from_labels(labels, 2, false);
            if (!cand) continue;
            if (conflicts_with_used(*cand, {})) continue;
            int cost = (int)cand->edges[0].size() + (int)cand->edges[1].size();
            uint64_t key = splitmix64(((uint64_t)(uint32_t)a << 32) ^ (uint32_t)b ^
                                      ((uint64_t)(uint32_t)cost * 0x9e3779b97f4a7c15ULL));
            scored.push_back(Scored{cost, key, a, b, std::move(*cand)});
        }
        if (scored.empty()) return 0;

        auto greedy_plan = [&](vector<int> order) {
            vector<unsigned char> leaf_used(n, 0);
            array<vector<unsigned char>, 2> edge_used;
            edge_used[0].assign(T[0].child.size(), 0);
            edge_used[1].assign(T[1].child.size(), 0);
            vector<int> plan;
            long long total_cost = 0;
            for (int idx : order) {
                if (deadline::expired()) break;
                const Scored& s = scored[idx];
                if (leaf_used[s.a] || leaf_used[s.b]) continue;
                bool conflict = false;
                for (int ti = 0; ti < 2 && !conflict; ++ti) {
                    for (int e : s.comp.edges[ti]) {
                        if (edge_used[ti][e]) { conflict = true; break; }
                    }
                }
                if (conflict) continue;
                leaf_used[s.a] = leaf_used[s.b] = 1;
                for (int ti = 0; ti < 2; ++ti) for (int e : s.comp.edges[ti]) edge_used[ti][e] = 1;
                plan.push_back(idx);
                total_cost += s.cost;
            }
            return pair<vector<int>, long long>(std::move(plan), total_cost);
        };

        vector<int> base_order(scored.size());
        iota(base_order.begin(), base_order.end(), 0);
        vector<int> best_plan;
        long long best_cost = numeric_limits<long long>::max();
        auto offer_order = [&](vector<int> order) {
            auto [plan, cost] = greedy_plan(std::move(order));
            if (plan.size() > best_plan.size() ||
                (plan.size() == best_plan.size() && cost < best_cost)) {
                best_cost = cost;
                best_plan = std::move(plan);
            }
        };

        for (int variant = 0; variant < 8 && !deadline::expired(); ++variant) {
            vector<int> order = base_order;
            sort(order.begin(), order.end(), [&](int ia, int ib) {
                const Scored& x = scored[ia];
                const Scored& y = scored[ib];
                if (variant == 0) {
                    if (x.cost != y.cost) return x.cost < y.cost;
                    if (x.a != y.a) return x.a < y.a;
                    return x.b < y.b;
                }
                if (variant == 1) {
                    if (x.cost / 4 != y.cost / 4) return x.cost / 4 < y.cost / 4;
                    return x.key < y.key;
                }
                if (variant == 2) {
                    if (x.cost != y.cost) return x.cost > y.cost;
                    return x.key < y.key;
                }
                uint64_t sx = x.key ^ splitmix64((uint64_t)variant * 0x517cc1b727220a95ULL + (uint64_t)x.cost);
                uint64_t sy = y.key ^ splitmix64((uint64_t)variant * 0x517cc1b727220a95ULL + (uint64_t)y.cost);
                return sx < sy;
            });
            offer_order(std::move(order));
        }

        int added = 0;
        for (int idx : best_plan) {
            if (deadline::expired()) break;
            auto& s = scored[idx];
            int ca = label_to_comp[s.a], cb = label_to_comp[s.b];
            if (ca < 0 || cb < 0 || ca == cb) continue;
            if (!comps[ca].active || !comps[cb].active || comps[ca].size != 1 || comps[cb].size != 1) continue;
            if (try_replace({ca, cb}, std::move(s.comp))) ++added;
        }
        return added;
    }

    vector<vector<int>> singleton_pack_candidates_from_tree(int ti, int min_sz, int max_sz) {
        const Tree& tr = T[ti];
        vector<vector<int>> bucket(tr.child.size());
        vector<unsigned char> bad(tr.child.size(), 0);
        vector<vector<int>> cands;
        for (int u : tr.post) {
            if (tr.child[u][0] == -1) {
                int lab = tr.leaf_label[u];
                int cid = label_to_comp[lab];
                if (cid >= 0 && comps[cid].active && comps[cid].size == 1) bucket[u].push_back(lab);
            } else {
                vector<int> v;
                bool too_big = false;
                for (int ch : tr.child[u]) {
                    if (bad[ch]) { too_big = true; break; }
                    v.insert(v.end(), bucket[ch].begin(), bucket[ch].end());
                    if ((int)v.size() > max_sz) { too_big = true; break; }
                }
                if (too_big) { bad[u] = 1; continue; }
                sort(v.begin(), v.end());
                bucket[u] = std::move(v);
                if ((int)bucket[u].size() >= min_sz && (int)bucket[u].size() <= max_sz) cands.push_back(bucket[u]);
            }
        }
        return cands;
    }

    vector<vector<int>> singleton_chain_window_candidates(int max_sz) {
        vector<vector<int>> all;
        for (int ti = 0; ti < 2; ++ti) {
            vector<int> active_order;
            active_order.reserve(n);
            for (int lab : T[ti].leaf_order) {
                int cid = label_to_comp[lab];
                if (cid >= 0 && comps[cid].active && comps[cid].size == 1) active_order.push_back(lab);
            }
            for (int i = 0; i < (int)active_order.size(); ++i) {
                vector<int> w;
                w.reserve(max_sz);
                for (int j = i; j < (int)active_order.size() && j < i + max_sz; ++j) {
                    w.push_back(active_order[j]);
                    if ((int)w.size() >= PACK_MIN) {
                        vector<int> s = w;
                        sort(s.begin(), s.end());
                        all.push_back(std::move(s));
                    }
                }
            }
        }
        return all;
    }

    int singleton_pack_pass(
        int max_pack_size = PACK_MAX,
        int max_bb_candidates = SINGLETON_PACK_BB_MAX_CANDIDATES,
        int max_bb_nodes = SINGLETON_PACK_BB_MAX_NODES
    ) {
        vector<vector<int>> raw;
        for (int ti = 0; ti < 2; ++ti) {
            auto v = singleton_pack_candidates_from_tree(ti, PACK_MIN, max_pack_size);
            raw.insert(raw.end(), v.begin(), v.end());
        }
        auto chains = singleton_chain_window_candidates(max_pack_size);
        raw.insert(raw.end(), chains.begin(), chains.end());
        sort(raw.begin(), raw.end());
        raw.erase(unique(raw.begin(), raw.end()), raw.end());

        struct Scored { int saved; int cost; vector<int> labels; Component comp; };
        vector<Scored> scored;
        scored.reserve(raw.size());
        for (auto labels : raw) {
            bool ok = true;
            for (int lab : labels) {
                int cid = label_to_comp[lab];
                if (cid < 0 || !comps[cid].active || comps[cid].size != 1) { ok = false; break; }
            }
            if (!ok) continue;
            auto cand = build_component_from_labels(labels, SMALL_RESTRICT_LIMIT, false);
            if (!cand) continue;
            if (conflicts_with_used(*cand, {})) continue;
            int cost = (int)cand->edges[0].size() + (int)cand->edges[1].size();
            scored.push_back({(int)labels.size() - 1, cost, std::move(labels), std::move(*cand)});
        }
        sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
            if (a.saved != b.saved) return a.saved > b.saved;
            if (a.cost != b.cost) return a.cost < b.cost;
            return a.labels < b.labels;
        });

        auto can_take = [&](int idx,
                            const vector<unsigned char>& leaf_used,
                            const array<vector<unsigned char>, 2>& edge_used) -> bool {
            const Scored& s = scored[idx];
            for (int lab : s.labels) if (leaf_used[lab]) return false;
            for (int ti = 0; ti < 2; ++ti) {
                for (int e : s.comp.edges[ti]) if (edge_used[ti][e]) return false;
            }
            return true;
        };
        auto mark_take = [&](int idx,
                             vector<unsigned char>& leaf_used,
                             array<vector<unsigned char>, 2>& edge_used,
                             bool val) {
            const Scored& s = scored[idx];
            for (int lab : s.labels) leaf_used[lab] = (unsigned char)val;
            for (int ti = 0; ti < 2; ++ti) {
                for (int e : s.comp.edges[ti]) edge_used[ti][e] = (unsigned char)val;
            }
        };
        auto extend_greedily = [&](const vector<int>& forced) {
            vector<int> plan;
            vector<unsigned char> forced_idx(scored.size(), 0);
            vector<unsigned char> leaf_used(n, 0);
            array<vector<unsigned char>, 2> edge_used;
            edge_used[0].assign(T[0].child.size(), 0);
            edge_used[1].assign(T[1].child.size(), 0);
            int score = 0;
            for (int idx : forced) {
                if (idx < 0 || idx >= (int)scored.size()) continue;
                if (forced_idx[idx]) continue;
                if (!can_take(idx, leaf_used, edge_used)) continue;
                forced_idx[idx] = 1;
                mark_take(idx, leaf_used, edge_used, true);
                score += scored[idx].saved;
                plan.push_back(idx);
            }
            for (int idx = 0; idx < (int)scored.size(); ++idx) {
                if (deadline::expired()) break;
                if (forced_idx[idx]) continue;
                if (!can_take(idx, leaf_used, edge_used)) continue;
                mark_take(idx, leaf_used, edge_used, true);
                score += scored[idx].saved;
                plan.push_back(idx);
            }
            return pair<int, vector<int>>(score, std::move(plan));
        };

        auto best_plan = extend_greedily({});
        if (!deadline::expired() && deadline::seconds_left() > 2.0 && !scored.empty()) {
            int pool = min((int)scored.size(), max_bb_candidates);
            vector<int> suffix_saved(pool + 1, 0);
            for (int i = pool - 1; i >= 0; --i) suffix_saved[i] = suffix_saved[i + 1] + scored[i].saved;

            vector<unsigned char> bb_leaf_used(n, 0);
            array<vector<unsigned char>, 2> bb_edge_used;
            bb_edge_used[0].assign(T[0].child.size(), 0);
            bb_edge_used[1].assign(T[1].child.size(), 0);
            vector<int> cur_branch, best_branch;
            int cur_score = 0, best_branch_score = 0, search_nodes = 0;

            // Start with a greedy solution inside the bounded pool so the DFS
            // has a meaningful incumbent for pruning.
            for (int idx = 0; idx < pool; ++idx) {
                if (!can_take(idx, bb_leaf_used, bb_edge_used)) continue;
                mark_take(idx, bb_leaf_used, bb_edge_used, true);
                best_branch_score += scored[idx].saved;
                best_branch.push_back(idx);
            }
            fill(bb_leaf_used.begin(), bb_leaf_used.end(), 0);
            fill(bb_edge_used[0].begin(), bb_edge_used[0].end(), 0);
            fill(bb_edge_used[1].begin(), bb_edge_used[1].end(), 0);

            function<void(int)> dfs = [&](int idx) {
                if (deadline::expired()) return;
                if (++search_nodes > max_bb_nodes) return;
                if (cur_score + suffix_saved[idx] <= best_branch_score) return;
                if (idx >= pool) {
                    if (cur_score > best_branch_score) {
                        best_branch_score = cur_score;
                        best_branch = cur_branch;
                    }
                    return;
                }
                if (can_take(idx, bb_leaf_used, bb_edge_used)) {
                    mark_take(idx, bb_leaf_used, bb_edge_used, true);
                    cur_branch.push_back(idx);
                    cur_score += scored[idx].saved;
                    dfs(idx + 1);
                    cur_score -= scored[idx].saved;
                    cur_branch.pop_back();
                    mark_take(idx, bb_leaf_used, bb_edge_used, false);
                }
                dfs(idx + 1);
            };
            dfs(0);

            auto bb_plan = extend_greedily(best_branch);
            if (bb_plan.first > best_plan.first) best_plan = std::move(bb_plan);
        }

        int added = 0;
        for (int idx : best_plan.second) {
            if (deadline::expired()) break;
            auto& s = scored[idx];
            vector<int> old_ids;
            bool ok = true;
            for (int lab : s.labels) {
                int cid = label_to_comp[lab];
                if (cid < 0 || !comps[cid].active || comps[cid].size != 1) { ok = false; break; }
                old_ids.push_back(cid);
            }
            if (!ok) continue;
            sort(old_ids.begin(), old_ids.end());
            old_ids.erase(unique(old_ids.begin(), old_ids.end()), old_ids.end());
            if ((int)old_ids.size() != (int)s.labels.size()) continue;
            if (try_replace(old_ids, std::move(s.comp))) ++added;
        }
        return added;
    }

    vector<vector<int>> component_pack_windows_from_tree(int ti) {
        const Tree& tr = T[ti];
        vector<vector<int>> ids(tr.child.size());
        vector<int> leaves(tr.child.size(), 0);
        vector<unsigned char> bad(tr.child.size(), 0);
        vector<vector<int>> wins;
        for (int u : tr.post) {
            if (tr.child[u][0] == -1) {
                int lab = tr.leaf_label[u];
                int cid = label_to_comp[lab];
                if (cid >= 0 && comps[cid].active) {
                    ids[u] = {cid};
                    leaves[u] = comps[cid].size;
                }
            } else {
                vector<int> v;
                bool too = false;
                int total = 0;
                for (int ch : tr.child[u]) {
                    if (bad[ch]) { too = true; break; }
                    v.insert(v.end(), ids[ch].begin(), ids[ch].end());
                }
                sort(v.begin(), v.end());
                v.erase(unique(v.begin(), v.end()), v.end());
                for (int id : v) total += comps[id].size;
                if ((int)v.size() > COMPONENT_PACK_MAX_COMPONENTS || total > COMPONENT_PACK_MAX_LEAVES) too = true;
                if (too) { bad[u] = 1; continue; }
                ids[u] = std::move(v);
                leaves[u] = total;
                if ((int)ids[u].size() >= 2 && leaves[u] <= COMPONENT_PACK_MAX_LEAVES) wins.push_back(ids[u]);
            }
        }
        return wins;
    }

    vector<vector<int>> component_pack_windows() {
        vector<vector<int>> wins;
        for (int ti = 0; ti < 2; ++ti) {
            auto w = component_pack_windows_from_tree(ti);
            wins.insert(wins.end(), w.begin(), w.end());
        }
        for (int ti = 0; ti < 2; ++ti) {
            vector<int> order;
            int last = -1;
            for (int lab : T[ti].leaf_order) {
                int cid = label_to_comp[lab];
                if (cid >= 0 && comps[cid].active && cid != last) {
                    order.push_back(cid);
                    last = cid;
                }
            }
            for (int i = 0; i < (int)order.size(); ++i) {
                vector<int> ids;
                int leaves = 0;
                for (int j = i; j < (int)order.size() && (int)ids.size() < COMPONENT_PACK_MAX_COMPONENTS; ++j) {
                    int id = order[j];
                    if (find(ids.begin(), ids.end(), id) != ids.end()) continue;
                    ids.push_back(id);
                    leaves += comps[id].size;
                    if (leaves > COMPONENT_PACK_MAX_LEAVES) break;
                    if ((int)ids.size() >= 2) {
                        vector<int> s = ids;
                        sort(s.begin(), s.end());
                        wins.push_back(std::move(s));
                    }
                }
            }
        }
        sort(wins.begin(), wins.end());
        wins.erase(unique(wins.begin(), wins.end()), wins.end());
        sort(wins.begin(), wins.end(), [&](const vector<int>& a, const vector<int>& b) {
            int sa = 0, sb = 0;
            for (int id : a) sa += comps[id].size;
            for (int id : b) sb += comps[id].size;
            if (a.size() != b.size()) return a.size() > b.size();
            if (sa != sb) return sa > sb;
            return a < b;
        });
        if ((int)wins.size() > COMPONENT_PACK_MAX_CANDIDATES) wins.resize(COMPONENT_PACK_MAX_CANDIDATES);
        return wins;
    }

    vector<vector<int>> corridor_windows_from_tree(int ti) {
        const Tree& tr = T[ti];
        vector<vector<int>> ids(tr.child.size());
        vector<int> leaves(tr.child.size(), 0);
        vector<unsigned char> bad(tr.child.size(), 0);
        vector<vector<int>> wins;
        for (int u : tr.post) {
            if (tr.child[u][0] == -1) {
                int lab = tr.leaf_label[u];
                int cid = label_to_comp[lab];
                if (cid >= 0 && comps[cid].active) {
                    ids[u] = {cid};
                    leaves[u] = comps[cid].size;
                }
            } else {
                vector<int> v;
                bool too = false;
                int total = 0;
                for (int ch : tr.child[u]) {
                    if (bad[ch]) { too = true; break; }
                    v.insert(v.end(), ids[ch].begin(), ids[ch].end());
                }
                sort(v.begin(), v.end());
                v.erase(unique(v.begin(), v.end()), v.end());
                for (int id : v) total += comps[id].size;
                if ((int)v.size() > CORRIDOR_WINDOW_MAX_COMPONENTS || total > CORRIDOR_WINDOW_MAX_LEAVES) too = true;
                if (too) { bad[u] = 1; continue; }
                ids[u] = std::move(v);
                leaves[u] = total;
                if ((int)ids[u].size() >= 2 && leaves[u] <= CORRIDOR_WINDOW_MAX_LEAVES) wins.push_back(ids[u]);
            }
        }
        return wins;
    }

    vector<vector<int>> corridor_windows() {
        vector<vector<int>> wins;
        for (int ti = 0; ti < 2; ++ti) {
            auto w = corridor_windows_from_tree(ti);
            wins.insert(wins.end(), w.begin(), w.end());
        }
        for (int ti = 0; ti < 2; ++ti) {
            vector<int> order;
            int last = -1;
            for (int lab : T[ti].leaf_order) {
                int cid = label_to_comp[lab];
                if (cid >= 0 && comps[cid].active && cid != last) {
                    order.push_back(cid);
                    last = cid;
                }
            }
            for (int i = 0; i < (int)order.size(); ++i) {
                vector<int> ids;
                int leaves = 0;
                for (int j = i; j < (int)order.size() && (int)ids.size() < CORRIDOR_WINDOW_MAX_COMPONENTS; ++j) {
                    int id = order[j];
                    if (find(ids.begin(), ids.end(), id) != ids.end()) continue;
                    ids.push_back(id);
                    leaves += comps[id].size;
                    if (leaves > CORRIDOR_WINDOW_MAX_LEAVES) break;
                    if ((int)ids.size() >= 2) {
                        vector<int> s = ids;
                        sort(s.begin(), s.end());
                        wins.push_back(std::move(s));
                    }
                }
            }
        }
        sort(wins.begin(), wins.end());
        wins.erase(unique(wins.begin(), wins.end()), wins.end());
        sort(wins.begin(), wins.end(), [&](const vector<int>& a, const vector<int>& b) {
            int sa = 0, sb = 0;
            for (int id : a) sa += comps[id].size;
            for (int id : b) sb += comps[id].size;
            if (a.size() != b.size()) return a.size() > b.size();
            if (sa != sb) return sa > sb;
            return a < b;
        });
        if ((int)wins.size() > CORRIDOR_WINDOW_MAX_CANDIDATES) wins.resize(CORRIDOR_WINDOW_MAX_CANDIDATES);
        return wins;
    }

    int component_pack_pass() {
        auto raw = component_pack_windows();
        struct Scored { int saved; int leaves; int cost; vector<int> ids; vector<int> labels; Component comp; };
        vector<Scored> scored;
        scored.reserve(min((int)raw.size(), COMPONENT_PACK_MAX_CANDIDATES));
        for (const auto& ids0 : raw) {
            if (deadline::expired()) break;
            vector<int> ids;
            for (int id : ids0) if (id >= 0 && id < (int)comps.size() && comps[id].active) ids.push_back(id);
            sort(ids.begin(), ids.end());
            ids.erase(unique(ids.begin(), ids.end()), ids.end());
            if ((int)ids.size() < 2 || (int)ids.size() > COMPONENT_PACK_MAX_COMPONENTS) continue;
            vector<int> labels;
            int leaf_count = 0;
            for (int id : ids) {
                leaf_count += comps[id].size;
                labels.insert(labels.end(), comps[id].labels.begin(), comps[id].labels.end());
            }
            if (leaf_count > COMPONENT_PACK_MAX_LEAVES) continue;
            sort(labels.begin(), labels.end());
            labels.erase(unique(labels.begin(), labels.end()), labels.end());
            if ((int)labels.size() != leaf_count) continue;
            auto cand = build_component_from_labels(labels, COMPONENT_PACK_MAX_LEAVES, true);
            if (!cand) continue;
            if ((int)cand->labels.size() <= 1) continue;
            if (conflicts_with_used(*cand, ids)) continue;
            int cost = (int)cand->edges[0].size() + (int)cand->edges[1].size();
            scored.push_back({(int)ids.size() - 1, (int)labels.size(), cost, std::move(ids), std::move(labels), std::move(*cand)});
        }
        sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
            if (a.saved != b.saved) return a.saved > b.saved;
            if (a.leaves != b.leaves) return a.leaves > b.leaves;
            if (a.cost != b.cost) return a.cost < b.cost;
            return a.labels < b.labels;
        });

        auto can_take = [&](int idx,
                            const vector<unsigned char>& comp_used,
                            const array<vector<unsigned char>, 2>& edge_used) -> bool {
            const Scored& s = scored[idx];
            for (int id : s.ids) {
                if (id < 0 || id >= (int)comp_used.size() || comp_used[id]) return false;
            }
            for (int ti = 0; ti < 2; ++ti) {
                for (int e : s.comp.edges[ti]) if (edge_used[ti][e]) return false;
            }
            return true;
        };
        auto mark_take = [&](int idx,
                             vector<unsigned char>& comp_used,
                             array<vector<unsigned char>, 2>& edge_used,
                             bool val) {
            const Scored& s = scored[idx];
            for (int id : s.ids) comp_used[id] = (unsigned char)val;
            for (int ti = 0; ti < 2; ++ti) {
                for (int e : s.comp.edges[ti]) edge_used[ti][e] = (unsigned char)val;
            }
        };
        auto extend_greedily = [&](const vector<int>& forced) {
            vector<int> plan;
            vector<unsigned char> forced_idx(scored.size(), 0);
            vector<unsigned char> comp_used(comps.size(), 0);
            array<vector<unsigned char>, 2> edge_used;
            edge_used[0].assign(T[0].child.size(), 0);
            edge_used[1].assign(T[1].child.size(), 0);
            int score = 0;
            for (int idx : forced) {
                if (idx < 0 || idx >= (int)scored.size()) continue;
                if (forced_idx[idx]) continue;
                if (!can_take(idx, comp_used, edge_used)) continue;
                forced_idx[idx] = 1;
                mark_take(idx, comp_used, edge_used, true);
                score += scored[idx].saved;
                plan.push_back(idx);
            }
            for (int idx = 0; idx < (int)scored.size(); ++idx) {
                if (deadline::expired()) break;
                if (forced_idx[idx]) continue;
                if (!can_take(idx, comp_used, edge_used)) continue;
                mark_take(idx, comp_used, edge_used, true);
                score += scored[idx].saved;
                plan.push_back(idx);
            }
            return pair<int, vector<int>>(score, std::move(plan));
        };

        auto best_plan = extend_greedily({});
        if (!deadline::expired() && deadline::seconds_left() > 2.0 && !scored.empty()) {
            int pool = min((int)scored.size(), COMPONENT_PACK_BB_MAX_CANDIDATES);
            vector<int> suffix_saved(pool + 1, 0);
            for (int i = pool - 1; i >= 0; --i) suffix_saved[i] = suffix_saved[i + 1] + scored[i].saved;

            vector<unsigned char> bb_comp_used(comps.size(), 0);
            array<vector<unsigned char>, 2> bb_edge_used;
            bb_edge_used[0].assign(T[0].child.size(), 0);
            bb_edge_used[1].assign(T[1].child.size(), 0);
            vector<int> cur_branch, best_branch;
            int cur_score = 0, best_branch_score = 0, search_nodes = 0;

            for (int idx = 0; idx < pool; ++idx) {
                if (!can_take(idx, bb_comp_used, bb_edge_used)) continue;
                mark_take(idx, bb_comp_used, bb_edge_used, true);
                best_branch_score += scored[idx].saved;
                best_branch.push_back(idx);
            }
            fill(bb_comp_used.begin(), bb_comp_used.end(), 0);
            fill(bb_edge_used[0].begin(), bb_edge_used[0].end(), 0);
            fill(bb_edge_used[1].begin(), bb_edge_used[1].end(), 0);

            function<void(int)> dfs = [&](int idx) {
                if (deadline::expired()) return;
                if (++search_nodes > COMPONENT_PACK_BB_MAX_NODES) return;
                if (cur_score + suffix_saved[idx] <= best_branch_score) return;
                if (idx >= pool) {
                    if (cur_score > best_branch_score) {
                        best_branch_score = cur_score;
                        best_branch = cur_branch;
                    }
                    return;
                }
                if (can_take(idx, bb_comp_used, bb_edge_used)) {
                    mark_take(idx, bb_comp_used, bb_edge_used, true);
                    cur_branch.push_back(idx);
                    cur_score += scored[idx].saved;
                    dfs(idx + 1);
                    cur_score -= scored[idx].saved;
                    cur_branch.pop_back();
                    mark_take(idx, bb_comp_used, bb_edge_used, false);
                }
                dfs(idx + 1);
            };
            dfs(0);

            auto bb_plan = extend_greedily(best_branch);
            if (bb_plan.first > best_plan.first) best_plan = std::move(bb_plan);
        }

        int merged = 0;
        for (int idx : best_plan.second) {
            if (deadline::expired()) break;
            auto& s = scored[idx];
            vector<int> ids;
            bool ok = true;
            for (int lab : s.labels) {
                int cid = label_to_comp[lab];
                if (cid < 0 || !comps[cid].active) { ok = false; break; }
                ids.push_back(cid);
            }
            if (!ok) continue;
            sort(ids.begin(), ids.end());
            ids.erase(unique(ids.begin(), ids.end()), ids.end());
            if ((int)ids.size() < 2 || (int)ids.size() > COMPONENT_PACK_MAX_COMPONENTS) continue;
            vector<int> current_labels;
            for (int id : ids) current_labels.insert(current_labels.end(), comps[id].labels.begin(), comps[id].labels.end());
            sort(current_labels.begin(), current_labels.end());
            current_labels.erase(unique(current_labels.begin(), current_labels.end()), current_labels.end());
            if (current_labels != s.labels) continue;
            if (try_replace(ids, std::move(s.comp))) ++merged;
        }
        return merged;
    }

    int corridor_merge_pass() {
        auto windows = (deadline::budget_seconds() >= 10.0 ? corridor_windows() : component_pack_windows());
        int merged = 0;
        int inspected = 0;
        for (const auto& win0 : windows) {
            if (deadline::expired()) break;
            if (++inspected > CORRIDOR_MERGE_MAX_WINDOWS) break;
            vector<int> ids;
            ids.reserve(win0.size());
            for (int id : win0) {
                if (id >= 0 && id < (int)comps.size() && comps[id].active) ids.push_back(id);
            }
            sort(ids.begin(), ids.end(), [&](int a, int b) {
                if (comps[a].size != comps[b].size) return comps[a].size > comps[b].size;
                return a < b;
            });
            ids.erase(unique(ids.begin(), ids.end()), ids.end());
            if ((int)ids.size() < 3) continue;
            if ((int)ids.size() > CORRIDOR_MERGE_MAX_IDS) ids.resize(CORRIDOR_MERGE_MAX_IDS);

            vector<int> chosen;
            vector<int> labels;
            Component best_comp;
            bool have_best = false;
            int best_saved = 0;

            for (int seed_pos = 0; seed_pos < (int)ids.size(); ++seed_pos) {
                if (deadline::expired()) break;
                vector<int> cur_ids{ids[seed_pos]};
                vector<int> cur_labels = comps[ids[seed_pos]].labels;
                Component cur_best;
                bool cur_have = false;

                bool changed = true;
                while (changed && !deadline::expired()) {
                    changed = false;
                    int take_id = -1;
                    Component take_comp;
                    vector<int> take_labels;
                    int take_gain = 0;

                    for (int id : ids) {
                        if (find(cur_ids.begin(), cur_ids.end(), id) != cur_ids.end()) continue;
                        vector<int> trial = cur_labels;
                        trial.insert(trial.end(), comps[id].labels.begin(), comps[id].labels.end());
                        sort(trial.begin(), trial.end());
                        trial.erase(unique(trial.begin(), trial.end()), trial.end());
                        if ((int)trial.size() > CORRIDOR_MERGE_MAX_LEAVES) continue;
                        auto cand = build_component_from_labels(trial, CORRIDOR_MERGE_MAX_LEAVES, true);
                        if (!cand) continue;
                        vector<int> exclude = cur_ids;
                        exclude.push_back(id);
                        if (conflicts_with_used(*cand, exclude)) continue;
                        int gain = (int)exclude.size() - 1;
                        if (gain > take_gain ||
                            (gain == take_gain && (take_id < 0 || comps[id].size > comps[take_id].size))) {
                            take_gain = gain;
                            take_id = id;
                            take_comp = std::move(*cand);
                            take_labels = std::move(trial);
                        }
                    }

                    if (take_id >= 0) {
                        cur_ids.push_back(take_id);
                        cur_labels = std::move(take_labels);
                        cur_best = std::move(take_comp);
                        cur_have = true;
                        changed = true;
                    }
                }

                int saved = (int)cur_ids.size() - 1;
                if (cur_have && saved > best_saved) {
                    best_saved = saved;
                    chosen = std::move(cur_ids);
                    labels = std::move(cur_labels);
                    best_comp = std::move(cur_best);
                    have_best = true;
                }
            }

            if (!have_best || best_saved <= 1) continue;
            sort(chosen.begin(), chosen.end());
            chosen.erase(unique(chosen.begin(), chosen.end()), chosen.end());
            bool ok = true;
            vector<int> current_labels;
            for (int id : chosen) {
                if (id < 0 || id >= (int)comps.size() || !comps[id].active) { ok = false; break; }
                current_labels.insert(current_labels.end(), comps[id].labels.begin(), comps[id].labels.end());
            }
            if (!ok) continue;
            sort(current_labels.begin(), current_labels.end());
            current_labels.erase(unique(current_labels.begin(), current_labels.end()), current_labels.end());
            if (current_labels != labels) continue;
            if (try_replace(chosen, std::move(best_comp))) ++merged;
        }
        return merged;
    }

    bool repair_component_window(const vector<int>& win_ids) {
        vector<int> active_ids;
        for (int id : win_ids) if (id >= 0 && id < (int)comps.size() && comps[id].active) active_ids.push_back(id);
        sort(active_ids.begin(), active_ids.end());
        active_ids.erase(unique(active_ids.begin(), active_ids.end()), active_ids.end());
        int m = (int)active_ids.size();
        if (m < 3 || m > COMPONENT_PACK_MAX_COMPONENTS) return false;

        int leaf_total = 0;
        for (int id : active_ids) leaf_total += comps[id].size;
        if (leaf_total > COMPONENT_PACK_MAX_LEAVES) return false;

        array<vector<unsigned char>,2> outside = used;
        for (int id : active_ids) {
            for (int ti = 0; ti < 2; ++ti) {
                for (int e : comps[id].edges[ti]) outside[ti][e] = 0;
            }
        }

        struct Cand { int mask; int pop; int cost; Component comp; };
        vector<Cand> cands;
        cands.reserve(min(COMPONENT_REPAIR_MAX_CANDIDATES, (1 << m)));
        int full = 1 << m;
        vector<int> labels;
        for (int mask = 1; mask < full; ++mask) {
            if (deadline::expired()) return false;
            int pop = __builtin_popcount((unsigned)mask);
            Component comp;
            if (pop == 1) {
                int bit = __builtin_ctz((unsigned)mask);
                comp = comps[active_ids[bit]];
            } else {
                labels.clear();
                int bits = mask;
                int leaves = 0;
                while (bits) {
                    int bit = bits & -bits;
                    int id = active_ids[__builtin_ctz((unsigned)bit)];
                    leaves += comps[id].size;
                    if (leaves > COMPONENT_PACK_MAX_LEAVES) break;
                    labels.insert(labels.end(), comps[id].labels.begin(), comps[id].labels.end());
                    bits ^= bit;
                }
                if (leaves > COMPONENT_PACK_MAX_LEAVES) continue;
                sort(labels.begin(), labels.end());
                labels.erase(unique(labels.begin(), labels.end()), labels.end());
                if ((int)labels.size() != leaves) continue;
                auto built = build_component_from_labels(labels, COMPONENT_PACK_MAX_LEAVES, true);
                if (!built) continue;
                comp = std::move(*built);
            }

            bool conflict = false;
            for (int ti = 0; ti < 2 && !conflict; ++ti) {
                for (int e : comp.edges[ti]) if (outside[ti][e]) { conflict = true; break; }
            }
            if (conflict) continue;
            int cost = (int)comp.edges[0].size() + (int)comp.edges[1].size();
            cands.push_back(Cand{mask, pop, cost, std::move(comp)});
            if ((int)cands.size() > COMPONENT_REPAIR_MAX_CANDIDATES) return false;
        }
        if (cands.empty()) return false;
        sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            if (a.pop != b.pop) return a.pop > b.pop;
            if (a.cost != b.cost) return a.cost < b.cost;
            return a.mask < b.mask;
        });

        vector<vector<int>> by_part(m);
        int max_pop = 1;
        for (int i = 0; i < (int)cands.size(); ++i) {
            max_pop = max(max_pop, cands[i].pop);
            int bits = cands[i].mask;
            while (bits) {
                int bit = bits & -bits;
                by_part[__builtin_ctz((unsigned)bit)].push_back(i);
                bits ^= bit;
            }
        }

        int best_count = m;
        vector<int> best;
        vector<int> chosen;
        array<vector<unsigned char>,2> local_used;
        local_used[0].assign(T[0].child.size(), 0);
        local_used[1].assign(T[1].child.size(), 0);
        int search_nodes = 0;

        function<void(int)> dfs = [&](int uncovered) {
            if (deadline::expired()) return;
            if (++search_nodes > COMPONENT_REPAIR_MAX_SEARCH_NODES) return;
            if ((int)chosen.size() >= best_count) return;
            if (uncovered == 0) {
                best = chosen;
                best_count = (int)chosen.size();
                return;
            }
            int lb = (__builtin_popcount((unsigned)uncovered) + max_pop - 1) / max_pop;
            if ((int)chosen.size() + lb >= best_count) return;

            int best_part = -1;
            vector<int> options;
            int rem = uncovered;
            while (rem) {
                int bit = rem & -rem;
                int part = __builtin_ctz((unsigned)bit);
                vector<int> opts;
                for (int ci : by_part[part]) {
                    const Cand& cand = cands[ci];
                    if ((cand.mask & uncovered) != cand.mask) continue;
                    bool conflict = false;
                    for (int ti = 0; ti < 2 && !conflict; ++ti) {
                        for (int e : cand.comp.edges[ti]) if (local_used[ti][e]) { conflict = true; break; }
                    }
                    if (!conflict) opts.push_back(ci);
                }
                if (opts.empty()) return;
                if (best_part < 0 || opts.size() < options.size()) {
                    best_part = part;
                    options = std::move(opts);
                }
                rem ^= bit;
            }

            for (int ci : options) {
                Cand& cand = cands[ci];
                for (int ti = 0; ti < 2; ++ti) for (int e : cand.comp.edges[ti]) local_used[ti][e] = 1;
                chosen.push_back(ci);
                dfs(uncovered ^ cand.mask);
                chosen.pop_back();
                for (int ti = 0; ti < 2; ++ti) for (int e : cand.comp.edges[ti]) local_used[ti][e] = 0;
                if (deadline::expired() || search_nodes > COMPONENT_REPAIR_MAX_SEARCH_NODES) return;
            }
        };
        dfs(full - 1);
        if (best.empty() || best_count >= m) return false;

        vector<Component> new_components;
        new_components.reserve(best.size());
        for (int ci : best) new_components.push_back(std::move(cands[ci].comp));
        for (int id : active_ids) remove_component(id);
        for (auto& c : new_components) add_component(std::move(c));
        verify_current_forest("repair_component_window");
        return true;
    }

    int component_repair_pass() {
        auto wins = component_pack_windows();
        int attempted = 0;
        for (const auto& w : wins) {
            if (deadline::expired()) break;
            if (++attempted > COMPONENT_REPAIR_MAX_WINDOWS) break;
            int before = active_count();
            if (repair_component_window(w)) return before - active_count();
        }
        return 0;
    }

    vector<vector<int>> local_windows_from_tree(int ti, int max_leaves, int max_components) {
        const Tree& tr = T[ti];
        vector<vector<int>> ids(tr.child.size());
        vector<int> leaf_total(tr.child.size(), 0);
        vector<unsigned char> bad(tr.child.size(), 0);
        vector<vector<int>> windows;
        for (int u : tr.post) {
            if (deadline::expired()) break;
            if (tr.child[u][0] == -1) {
                int lab = tr.leaf_label[u];
                int cid = label_to_comp[lab];
                if (cid >= 0 && comps[cid].active) {
                    ids[u] = {cid};
                    leaf_total[u] = comps[cid].size;
                }
            } else {
                vector<int> v;
                int leaves = 0;
                bool too = false;
                for (int ch : tr.child[u]) {
                    if (bad[ch]) { too = true; break; }
                    v.insert(v.end(), ids[ch].begin(), ids[ch].end());
                }
                sort(v.begin(), v.end());
                v.erase(unique(v.begin(), v.end()), v.end());
                for (int id : v) leaves += comps[id].size;
                if ((int)v.size() > max_components || leaves > max_leaves) too = true;
                if (too) { bad[u] = 1; continue; }
                ids[u] = std::move(v);
                leaf_total[u] = leaves;
                if ((int)ids[u].size() >= 3 && leaf_total[u] <= max_leaves) windows.push_back(ids[u]);
            }
        }
        return windows;
    }

    vector<vector<int>> local_windows(int max_leaves, int max_components, int max_windows) {
        vector<vector<int>> wins;
        for (int ti = 0; ti < 2; ++ti) {
            if (deadline::expired()) break;
            auto w = local_windows_from_tree(ti, max_leaves, max_components);
            wins.insert(wins.end(), w.begin(), w.end());
        }
        // Sliding windows over component order in leaf orders.
        for (int ti = 0; ti < 2; ++ti) {
            if (deadline::expired()) break;
            vector<int> order;
            int last = -1;
            for (int lab : T[ti].leaf_order) {
                int cid = label_to_comp[lab];
                if (cid >= 0 && comps[cid].active && cid != last) {
                    order.push_back(cid);
                    last = cid;
                }
            }
            for (int i = 0; i < (int)order.size(); ++i) {
                if (deadline::expired()) break;
                vector<int> ids;
                int leaves = 0;
                for (int j = i; j < (int)order.size() && (int)ids.size() < max_components; ++j) {
                    int id = order[j];
                    if (find(ids.begin(), ids.end(), id) != ids.end()) continue;
                    ids.push_back(id);
                    leaves += comps[id].size;
                    if (leaves > max_leaves) break;
                    if ((int)ids.size() >= 3) {
                        vector<int> s = ids;
                        sort(s.begin(), s.end());
                        wins.push_back(std::move(s));
                    }
                }
            }
        }
        if (deadline::expired()) return wins;
        sort(wins.begin(), wins.end());
        wins.erase(unique(wins.begin(), wins.end()), wins.end());
        sort(wins.begin(), wins.end(), [&](const vector<int>& a, const vector<int>& b) {
            int la = 0, lb = 0;
            for (int id : a) la += comps[id].size;
            for (int id : b) lb += comps[id].size;
            if (a.size() != b.size()) return a.size() > b.size();
            if (la != lb) return la < lb;
            return a < b;
        });
        if ((int)wins.size() > max_windows) wins.resize(max_windows);
        return wins;
    }

    bool repair_window(
        const vector<int>& win_ids,
        int max_leaves,
        int max_candidates,
        int max_search_nodes
    ) {
        vector<int> active_ids;
        for (int id : win_ids) if (id >= 0 && id < (int)comps.size() && comps[id].active) active_ids.push_back(id);
        sort(active_ids.begin(), active_ids.end());
        active_ids.erase(unique(active_ids.begin(), active_ids.end()), active_ids.end());
        if ((int)active_ids.size() < 3) return false;

        vector<int> labels;
        for (int id : active_ids) labels.insert(labels.end(), comps[id].labels.begin(), comps[id].labels.end());
        sort(labels.begin(), labels.end());
        labels.erase(unique(labels.begin(), labels.end()), labels.end());
        int w = (int)labels.size();
        if (w > max_leaves) return false;
        int current_count = (int)active_ids.size();

        // Outside-used masks: old window components are allowed to be replaced.
        array<vector<unsigned char>,2> outside = used;
        for (int id : active_ids) {
            for (int ti = 0; ti < 2; ++ti) for (int e : comps[id].edges[ti]) outside[ti][e] = 0;
        }

        struct Cand { int mask; int cost; Component comp; };
        vector<Cand> cands;
        int full = 1 << w;
        vector<int> subset_labels;
        subset_labels.reserve(w);
        for (int mask = 1; mask < full; ++mask) {
            if (deadline::expired()) return false;
            subset_labels.clear();
            int bits = mask;
            while (bits) {
                int bit = bits & -bits;
                subset_labels.push_back(labels[__builtin_ctz((unsigned)bit)]);
                bits ^= bit;
            }
            auto comp = build_component_from_labels(subset_labels, SMALL_RESTRICT_LIMIT, false);
            if (!comp) continue;
            bool conflict = false;
            for (int ti = 0; ti < 2 && !conflict; ++ti) {
                for (int e : comp->edges[ti]) if (outside[ti][e]) { conflict = true; break; }
            }
            if (conflict) continue;
            int cost = (int)comp->edges[0].size() + (int)comp->edges[1].size();
            cands.push_back({mask, cost, std::move(*comp)});
            if ((int)cands.size() > max_candidates) return false;
        }
        if (cands.empty()) return false;

        vector<vector<int>> by_leaf(w);
        for (int i = 0; i < (int)cands.size(); ++i) {
            int m = cands[i].mask;
            while (m) {
                int bit = m & -m;
                by_leaf[__builtin_ctz((unsigned)bit)].push_back(i);
                m ^= bit;
            }
        }
        for (auto& v : by_leaf) {
            sort(v.begin(), v.end(), [&](int a, int b) {
                int sa = __builtin_popcount((unsigned)cands[a].mask);
                int sb = __builtin_popcount((unsigned)cands[b].mask);
                if (sa != sb) return sa > sb;
                if (cands[a].cost != cands[b].cost) return cands[a].cost < cands[b].cost;
                return cands[a].mask < cands[b].mask;
            });
        }

        int best_count = current_count;
        vector<int> best;
        vector<int> chosen;
        int search_nodes = 0;
        int max_comp_size = 1;
        for (auto& c : cands) max_comp_size = max(max_comp_size, __builtin_popcount((unsigned)c.mask));
        array<vector<unsigned char>,2> local_used;
        local_used[0].assign(T[0].child.size(), 0);
        local_used[1].assign(T[1].child.size(), 0);

        function<void(int)> dfs = [&](int uncovered) {
            if (deadline::expired()) return;
            if (++search_nodes > max_search_nodes) return;
            if ((int)chosen.size() >= best_count) return;
            if (uncovered == 0) {
                best = chosen;
                best_count = (int)chosen.size();
                return;
            }
            int lb = (__builtin_popcount((unsigned)uncovered) + max_comp_size - 1) / max_comp_size;
            if ((int)chosen.size() + lb >= best_count) return;

            int best_leaf = -1;
            vector<int> options;
            int rem = uncovered;
            while (rem) {
                int bit = rem & -rem;
                int li = __builtin_ctz((unsigned)bit);
                vector<int> opts;
                for (int ci : by_leaf[li]) {
                    const Cand& cand = cands[ci];
                    if ((cand.mask & uncovered) != cand.mask) continue;
                    bool conflict = false;
                    for (int ti = 0; ti < 2 && !conflict; ++ti) {
                        for (int e : cand.comp.edges[ti]) if (local_used[ti][e]) { conflict = true; break; }
                    }
                    if (!conflict) opts.push_back(ci);
                }
                if (opts.empty()) return;
                if (best_leaf < 0 || opts.size() < options.size()) {
                    best_leaf = li;
                    options = std::move(opts);
                }
                rem ^= bit;
            }

            for (int ci : options) {
                Cand& cand = cands[ci];
                for (int ti = 0; ti < 2; ++ti) for (int e : cand.comp.edges[ti]) local_used[ti][e] = 1;
                chosen.push_back(ci);
                dfs(uncovered ^ cand.mask);
                chosen.pop_back();
                for (int ti = 0; ti < 2; ++ti) for (int e : cand.comp.edges[ti]) local_used[ti][e] = 0;
                if (deadline::expired() || search_nodes > max_search_nodes) return;
            }
        };
        dfs(full - 1);
        if (best.empty() || best_count >= current_count) return false;

        vector<Component> new_components;
        for (int ci : best) new_components.push_back(std::move(cands[ci].comp));
        for (int id : active_ids) remove_component(id);
        for (auto& c : new_components) add_component(std::move(c));
        return true;
    }

    int local_repair_pass(
        int max_leaves = LOCAL_MAX_LEAVES,
        int max_components = LOCAL_MAX_COMPONENTS,
        int max_windows = LOCAL_MAX_WINDOWS,
        int max_candidates = LOCAL_MAX_CANDIDATES,
        int max_search_nodes = LOCAL_MAX_SEARCH_NODES
    ) {
        auto wins = local_windows(max_leaves, max_components, max_windows);
        for (const auto& w : wins) {
            if (deadline::expired()) break;
            int before = active_count();
            if (repair_window(w, max_leaves, max_candidates, max_search_nodes)) return before - active_count();
        }
        return 0;
    }

    int deep_local_repair_pass() {
        if (deadline::expired() || deadline::seconds_left() < 20.0) return 0;
        if (deep_local_attempts_ >= DEEP_LOCAL_MAX_ATTEMPTS) return 0;
        ++deep_local_attempts_;
        return local_repair_pass(
            DEEP_LOCAL_MAX_LEAVES,
            DEEP_LOCAL_MAX_COMPONENTS,
            DEEP_LOCAL_MAX_WINDOWS,
            DEEP_LOCAL_MAX_CANDIDATES,
            DEEP_LOCAL_MAX_SEARCH_NODES
        );
    }

    // Conflict-directed repair. Greedy merging reaches a fixpoint with no
    // conflict-free merges left, yet ~1000 merges per large instance agree
    // topologically but are blocked by a third component. For each such blocked
    // merge (a,b) with a small blocker set, run the exact window re-partition on
    // {a,b} ∪ blockers: the optimizer can break up the blocker and form a∪b when
    // that yields fewer components. This is the conflict-resolution neighborhood
    // that pairwise/window merge passes cannot reach.
    int conflict_repair_pass(int max_leaves, int max_blockers, int max_windows,
                             int max_candidates, int max_search_nodes) {
        if (deadline::expired()) return 0;
        // Edge-ownership map (component id per tree edge), used only to select
        // windows; repair_window re-validates against the live `used` array, so
        // staleness after a repair affects efficiency, never correctness.
        array<vector<int>,2> owner;
        owner[0].assign(T[0].child.size(), -1);
        owner[1].assign(T[1].child.size(), -1);
        for (const auto& c : comps) if (c.active) {
            for (int ti = 0; ti < 2; ++ti) for (int e : c.edges[ti]) owner[ti][e] = c.id;
        }

        vector<pair<int,int>> pairs = component_pairs_from_tree(0);
        vector<pair<int,int>> p2 = component_pairs_from_tree(1);
        pairs.insert(pairs.end(), p2.begin(), p2.end());
        sort(pairs.begin(), pairs.end());
        pairs.erase(unique(pairs.begin(), pairs.end()), pairs.end());

        unordered_set<uint64_t> seen_window;
        int repaired = 0, windows = 0;
        for (auto [a, b] : pairs) {
            if (deadline::expired() || windows >= max_windows) break;
            if (a < 0 || b < 0 || a >= (int)comps.size() || b >= (int)comps.size()) continue;
            if (!comps[a].active || !comps[b].active) continue;
            if (comps[a].size + comps[b].size > max_leaves) continue;

            vector<int> labels;
            labels.insert(labels.end(), comps[a].labels.begin(), comps[a].labels.end());
            labels.insert(labels.end(), comps[b].labels.begin(), comps[b].labels.end());
            auto cand = build_component_from_labels(labels, MERGE_SMALL_LIMIT, true);
            if (!cand) continue;  // a∪b does not agree topologically

            vector<int> bl;
            bool too_many = false;
            for (int ti = 0; ti < 2 && !too_many; ++ti) {
                for (int e : cand->edges[ti]) {
                    int o = owner[ti][e];
                    if (o >= 0 && o != a && o != b &&
                        find(bl.begin(), bl.end(), o) == bl.end()) {
                        bl.push_back(o);
                        if ((int)bl.size() > max_blockers) { too_many = true; break; }
                    }
                }
            }
            if (too_many || bl.empty()) continue;  // empty => merge_pass already covers it

            vector<int> win = {a, b};
            int total = comps[a].size + comps[b].size;
            bool ok = true;
            for (int o : bl) {
                if (o < 0 || o >= (int)comps.size() || !comps[o].active) { ok = false; break; }
                total += comps[o].size;
                win.push_back(o);
            }
            if (!ok || total > max_leaves) continue;
            sort(win.begin(), win.end());
            win.erase(unique(win.begin(), win.end()), win.end());
            if ((int)win.size() < 3) continue;

            uint64_t h = 1469598103934665603ULL;
            for (int id : win) { h ^= (uint64_t)(uint32_t)id; h *= 1099511628211ULL; }
            if (!seen_window.insert(h).second) continue;
            ++windows;

            int before = active_count();
            if (repair_window(win, max_leaves, max_candidates, max_search_nodes)) {
                repaired += before - active_count();
                // Refresh ownership so subsequent windows see the new partition.
                owner[0].assign(T[0].child.size(), -1);
                owner[1].assign(T[1].child.size(), -1);
                for (const auto& c : comps) if (c.active) {
                    for (int ti = 0; ti < 2; ++ti) for (int e : c.edges[ti]) owner[ti][e] = c.id;
                }
            }
        }
        return repaired;
    }

    vector<Component> active_components_snapshot() const {
        vector<Component> out;
        out.reserve(active_count());
        for (const auto& c : comps) if (c.active) {
            Component x = c;
            x.id = -1;
            x.active = true;
            out.push_back(std::move(x));
        }
        return out;
    }

    bool seed_components_compatible(const vector<Component>& seed) const {
        vector<unsigned char> seen_label(n, 0);
        int label_total = 0;
        array<vector<unsigned char>,2> seen_edge;
        seen_edge[0].assign(T[0].child.size(), 0);
        seen_edge[1].assign(T[1].child.size(), 0);

        for (const Component& c : seed) {
            if (c.labels.empty() || c.size != (int)c.labels.size()) return false;
            label_total += c.size;
            for (int lab : c.labels) {
                if (lab < 0 || lab >= n || seen_label[lab]) return false;
                seen_label[lab] = 1;
            }
            for (int ti = 0; ti < 2; ++ti) {
                for (int e : c.edges[ti]) {
                    if (e < 0 || e >= (int)seen_edge[ti].size() || seen_edge[ti][e]) return false;
                    seen_edge[ti][e] = 1;
                }
            }
        }
        if (label_total != n) return false;
        for (int lab = 0; lab < n; ++lab) if (!seen_label[lab]) return false;
        return true;
    }

    void solve_subproblem_fast() {
        publish_singleton_fallback();
        if (solve_exact_small()) { publish_if_better(); return; }
        if (deadline::expired()) return;

        vector<Component> s0 = common_clade_sweep(0);
        if (deadline::expired()) return;
        vector<Component> s1 = common_clade_sweep(1);
        if (deadline::expired()) return;
        vector<Component> best = (s1.size() < s0.size() ? std::move(s1) : std::move(s0));
        install_components(std::move(best));
        publish_if_better();
        improvement_passes();
        publish_if_better();

        if (!deadline::expired() && n <= 3000 && deadline::seconds_left() > 4.0) {
            run_seed(pair_portfolio_seed(0));
        }
        if (!best_snapshot_.empty()) install_components(vector<Component>(best_snapshot_));
    }

    struct CommonClusterCandidate {
        int node0 = -1;
        int node1 = -1;
        int size = 0;
        int inside_parts = 0;
        int crossing = 0;
        int crossing_labels = 0;
        long long score = 0;
        vector<int> labels;
    };

    vector<CommonClusterCandidate> common_cluster_candidates(int variant) {
        unordered_map<LeafSetKey, int, LeafSetKeyHash> other;
        other.reserve(T[1].child.size() * 2 + 1);
        int max_size = n - 1;
        if (deadline::budget_seconds() < 60.0 || deadline::seconds_left() < 35.0) {
            max_size = min(max_size, 2500);
        } else {
            max_size = min(max_size, 7000);
        }

        for (int u = 0; u < (int)T[1].child.size(); ++u) {
            int sz = T[1].subtree_size[u];
            if (sz < COMMON_CLUSTER_MIN_SIZE || sz > max_size) continue;
            other.emplace(leaf_set_key(T[1].key[u]), u);
        }

        vector<CommonClusterCandidate> out;
        vector<unsigned char> in_cluster(n, 0);
        for (int u = 0; u < (int)T[0].child.size(); ++u) {
            if (deadline::expired()) break;
            int sz = T[0].subtree_size[u];
            if (sz < COMMON_CLUSTER_MIN_SIZE || sz > max_size) continue;
            auto it = other.find(leaf_set_key(T[0].key[u]));
            if (it == other.end()) continue;
            int v = it->second;
            if (T[0].key[u] == T[1].key[v]) continue;

            vector<int> labels = T[0].subtree_labels(u);
            for (int lab : labels) in_cluster[lab] = 1;
            int inside_parts = 0, crossing = 0, crossing_labels = 0;
            for (const Component& comp : comps) {
                if (!comp.active) continue;
                int inside = 0;
                for (int lab : comp.labels) inside += in_cluster[lab] ? 1 : 0;
                if (inside == 0) continue;
                if (inside == comp.size) {
                    ++inside_parts;
                } else {
                    ++crossing;
                    crossing_labels += min(inside, comp.size - inside);
                }
            }
            for (int lab : labels) in_cluster[lab] = 0;
            if (inside_parts < 4 && crossing == 0) continue;

            long long score = 0;
            int smaller_side = min(sz, n - sz);
            if ((variant & 1) == 0) {
                score = (long long)inside_parts * 1000000LL +
                        (long long)crossing * 100000LL +
                        (long long)smaller_side * 100LL - sz;
            } else {
                score = (long long)smaller_side * 1000000LL +
                        (long long)inside_parts * 1000LL +
                        (long long)crossing_labels * 10LL - crossing;
            }
            out.push_back(CommonClusterCandidate{u, v, sz, inside_parts, crossing,
                                                 crossing_labels, score, std::move(labels)});
        }

        sort(out.begin(), out.end(), [](const CommonClusterCandidate& a, const CommonClusterCandidate& b) {
            if (a.score != b.score) return a.score > b.score;
            if (a.inside_parts != b.inside_parts) return a.inside_parts > b.inside_parts;
            if (a.crossing != b.crossing) return a.crossing > b.crossing;
            if (a.size != b.size) return a.size > b.size;
            return a.node0 < b.node0;
        });
        if ((int)out.size() > COMMON_CLUSTER_SCAN_LIMIT) out.resize(COMMON_CLUSTER_SCAN_LIMIT);
        return out;
    }

    string top_with_cluster_marker_rec(
        int ti,
        int node,
        int cluster_node,
        const vector<int>& parent_to_local,
        int marker_label
    ) const {
        const Tree& tr = T[ti];
        if (node == cluster_node) {
            if (marker_label < 0) return string();
            return to_string(marker_label + 1);
        }
        if (tr.child[node][0] == -1) {
            int mapped = parent_to_local[tr.leaf_label[node]];
            if (mapped < 0) return string();
            return to_string(mapped + 1);
        }
        string a = top_with_cluster_marker_rec(ti, tr.child[node][0], cluster_node,
                                               parent_to_local, marker_label);
        string b = top_with_cluster_marker_rec(ti, tr.child[node][1], cluster_node,
                                               parent_to_local, marker_label);
        if (a.empty()) return b;
        if (b.empty()) return a;
        return "(" + a + "," + b + ")";
    }

    struct MappedSubproblemResult {
        bool ok = false;
        bool marker_found = false;
        vector<int> marker_labels;
        vector<Component> regular;
    };

    MappedSubproblemResult solve_mapped_subproblem(
        const vector<int>& local_to_parent,
        const string& nw0,
        const string& nw1,
        double slice_cap = 14.0,
        double slice_floor = 3.0,
        double min_remaining = 4.0,
        bool full_solver = false
    ) {
        MappedSubproblemResult out;
        if (deadline::expired() || deadline::seconds_left() < min_remaining) return out;

        // ---- Exact-core repair (never-regress): if this is a clean cluster
        // (no marker) small enough to solve EXACTLY within the node cap, use the
        // proven optimum. It is by construction <= any heuristic result, so we
        // return it directly and skip the heuristic sub-solve. If the B&B does
        // not finish (nullopt) we fall through to the heuristic below.
        {
            int m = (int)local_to_parent.size();
            bool has_marker = false;
            for (int p : local_to_parent) if (p < 0) { has_marker = true; break; }
            if (use_exact_core_ && !has_marker && m >= 2 && m <= EXACT_CORE_MAX_LEAVES) {
                auto exact = exactmaf::solve(nw0, nw1, m, EXACT_CORE_NODE_CAP, /*max_seconds=*/1.0);
                if (exact) {
                    MappedSubproblemResult eo;
                    eo.ok = true;
                    bool good = true;
                    for (auto& comp_local : *exact) {
                        vector<int> pls; pls.reserve(comp_local.size());
                        for (int lab : comp_local) {
                            if (lab < 0 || lab >= m) { good = false; break; }
                            pls.push_back(local_to_parent[lab]);   // all >=0 (no marker)
                        }
                        if (!good) break;
                        auto comp = build_agreement_component_unbounded(std::move(pls));
                        if (!comp) { good = false; break; }
                        eo.regular.push_back(std::move(*comp));
                    }
                    if (good && !eo.regular.empty()) return eo;
                }
            }
        }

        Solver sub((int)local_to_parent.size(), nw0, nw1, false);
        auto saved_deadline = deadline::current_deadline();
        double saved_budget = deadline::budget_seconds();
        double floor = min(slice_cap, slice_floor);
        double slice = min(slice_cap, max(floor, deadline::seconds_left() / 10.0));
        deadline::shorten_for(slice);
        // Let mapped subproblems use the same budget-gated tactics they would
        // use when run as standalone short-budget instances.
        deadline::set_budget_seconds(slice);
        if (full_solver) sub.solve();
        else sub.solve_subproblem_fast();
        deadline::set_budget_seconds(saved_budget);
        deadline::restore_deadline(saved_deadline);
        const vector<Component>& local_snapshot =
            (!sub.best_snapshot_.empty() ? sub.best_snapshot_ : sub.comps);

        out.ok = true;
        for (const Component& lc : local_snapshot) {
            if (!lc.active) continue;
            bool has_marker = false;
            vector<int> parent_labels;
            parent_labels.reserve(lc.labels.size());
            for (int lab : lc.labels) {
                if (lab < 0 || lab >= (int)local_to_parent.size()) {
                    out.ok = false;
                    return out;
                }
                int parent = local_to_parent[lab];
                if (parent < 0) has_marker = true;
                else parent_labels.push_back(parent);
            }
            if (has_marker) {
                if (out.marker_found) {
                    out.ok = false;
                    return out;
                }
                out.marker_found = true;
                sort(parent_labels.begin(), parent_labels.end());
                out.marker_labels = std::move(parent_labels);
            } else {
                auto comp = build_agreement_component_unbounded(std::move(parent_labels));
                if (!comp) {
                    out.ok = false;
                    return out;
                }
                out.regular.push_back(std::move(*comp));
            }
        }
        return out;
    }

    vector<Component> solve_closed_common_cluster_node(
        int node0,
        int node1,
        double slice_cap,
        double slice_floor = 0.35,
        bool full_solver = false
    ) {
        if (deadline::expired()) return {};
        vector<int> local_to_parent = T[0].subtree_labels(node0);
        vector<int> parent_to_local(n, -1);
        for (int i = 0; i < (int)local_to_parent.size(); ++i) {
            parent_to_local[local_to_parent[i]] = i;
        }

        string nw0 = T[0].subtree_newick_remap(node0, parent_to_local);
        string nw1 = T[1].subtree_newick_remap(node1, parent_to_local);
        nw0.push_back(';');
        nw1.push_back(';');

        MappedSubproblemResult res = solve_mapped_subproblem(
            local_to_parent, nw0, nw1, slice_cap, slice_floor, 1.25, full_solver);
        if (!res.ok || res.marker_found || res.regular.empty()) return {};
        return std::move(res.regular);
    }

    struct ClosedCommonClusterRepairCandidate {
        int node0 = -1;
        int node1 = -1;
        int size = 0;
        int inside_parts = 0;
        long long score = 0;
        vector<int> inside_ids;
    };

    int small_common_cluster_repair_pass() {
        if (deadline::expired() || deadline::seconds_left() < 2.0 || n < 500) return 0;
        if (small_common_cluster_attempts_ >= SMALL_COMMON_CLUSTER_REPAIR_MAX_ATTEMPTS) return 0;
        ++small_common_cluster_attempts_;
        if (!best_snapshot_.empty()) install_components(vector<Component>(best_snapshot_));
        bool tail_common_cluster_polish =
            final_common_cluster_polish_ ||
            (deadline::budget_seconds() >= 100.0 && deadline::seconds_left() < 35.0);

        const int min_size = 40;
        int max_size = 180;
        if (tail_common_cluster_polish && deadline::budget_seconds() >= 180.0 &&
            deadline::seconds_left() > 8.0) {
            max_size = 2200;
        } else if (tail_common_cluster_polish && deadline::budget_seconds() >= 60.0 &&
                   deadline::seconds_left() > 4.0) {
            max_size = 900;
        } else if (tail_common_cluster_polish && deadline::budget_seconds() >= 30.0 &&
                   deadline::seconds_left() > 6.0) {
            max_size = 420;
        }
        unordered_map<LeafSetKey, int, LeafSetKeyHash> other;
        other.reserve(T[1].child.size() * 2 + 1);
        for (int u = 0; u < (int)T[1].child.size(); ++u) {
            int sz = T[1].subtree_size[u];
            if (sz < min_size || sz > max_size) continue;
            other.emplace(leaf_set_key(T[1].key[u]), u);
        }

        vector<ClosedCommonClusterRepairCandidate> candidates;
        vector<int> touched;
        vector<int> hit_count(comps.size(), 0);
        for (int u = 0; u < (int)T[0].child.size(); ++u) {
            if (deadline::expired()) break;
            int sz = T[0].subtree_size[u];
            if (sz < min_size || sz > max_size) continue;
            auto it = other.find(leaf_set_key(T[0].key[u]));
            if (it == other.end()) continue;
            int v = it->second;
            if (T[0].key[u] == T[1].key[v]) continue;

            vector<int> labels = T[0].subtree_labels(u);
            touched.clear();
            for (int lab : labels) {
                int id = label_to_comp[lab];
                if (id < 0 || id >= (int)comps.size() || !comps[id].active) continue;
                if (hit_count[id] == 0) touched.push_back(id);
                ++hit_count[id];
            }

            vector<int> inside_ids;
            inside_ids.reserve(touched.size());
            int crossing = 0;
            for (int id : touched) {
                if (hit_count[id] == comps[id].size) inside_ids.push_back(id);
                else ++crossing;
                hit_count[id] = 0;
            }
            if (crossing != 0) continue;
            int inside_parts = (int)inside_ids.size();
            if (inside_parts < 12) continue;
            if (sz <= 180) {
                if (inside_parts * 4 < sz * 3) continue;
            } else {
                if (inside_parts < 30) continue;
                if (inside_parts * 5 < sz * 2) continue;
            }

            long long ratio_score = ((long long)inside_parts * 1000000LL) / max(1, sz);
            long long dense_small_bonus =
                (sz <= 180 && inside_parts >= 80 && inside_parts * 4 >= sz * 3)
                    ? 4000000000000000000LL : 0LL;
            long long score = dense_small_bonus +
                              (long long)inside_parts * 1000000000LL +
                              ratio_score * 1000LL - sz;
            candidates.push_back(ClosedCommonClusterRepairCandidate{
                u, v, sz, inside_parts, score, std::move(inside_ids)});
        }

        if (candidates.empty()) return 0;
        sort(candidates.begin(), candidates.end(),
             [](const ClosedCommonClusterRepairCandidate& a,
                const ClosedCommonClusterRepairCandidate& b) {
                 if (a.score != b.score) return a.score > b.score;
                 if (a.inside_parts != b.inside_parts) return a.inside_parts > b.inside_parts;
                 if (a.size != b.size) return a.size < b.size;
                 return a.node0 < b.node0;
             });

        int attempts = 0;
        double pass_budget = tail_common_cluster_polish
            ? min(10.0, max(3.5, deadline::seconds_left() / 3.0))
            : min(10.0, max(1.0, deadline::seconds_left() / 12.0));
        auto pass_deadline = deadline::clock_t_::now() +
                             std::chrono::milliseconds((long long)(pass_budget * 1000.0));
        int best_saved = 0;
        vector<Component> best_seed;
        for (const auto& cand : candidates) {
            if (deadline::expired() || deadline::seconds_left() < 1.25) break;
            if (deadline::clock_t_::now() >= pass_deadline) break;
            if (++attempts > (tail_common_cluster_polish ? 6 : 4)) break;

            double slice_cap = cand.size <= 180 ? 1.6 : (cand.size <= 500 ? 2.8 : 3.2);
            double slice_floor = cand.size <= 180 ? 0.35 : 2.2;
            bool full_solver = tail_common_cluster_polish && cand.size > 180;
            vector<Component> parts = solve_closed_common_cluster_node(
                cand.node0, cand.node1, slice_cap, slice_floor, full_solver);
            if (parts.empty() || (int)parts.size() >= cand.inside_parts) continue;
            int saved = cand.inside_parts - (int)parts.size();
            int min_saved = cand.size <= 180 ? 1 : max(8, cand.inside_parts / 32);
            if (saved < min_saved) continue;

            vector<unsigned char> remove(comps.size(), 0);
            for (int id : cand.inside_ids) {
                if (id < 0 || id >= (int)comps.size() || !comps[id].active) {
                    remove.clear();
                    break;
                }
                remove[id] = 1;
            }
            if (remove.empty()) continue;

            vector<Component> seed;
            seed.reserve(active_count() - cand.inside_parts + parts.size());
            for (const Component& c : comps) {
                if (c.active && !remove[c.id]) seed.push_back(c);
            }
            for (auto& c : parts) seed.push_back(std::move(c));
            if (!seed_components_compatible(seed)) continue;

            if (tail_common_cluster_polish) {
                if (saved > best_saved) {
                    best_saved = saved;
                    best_seed = std::move(seed);
                }
                continue;
            }
            install_components(std::move(seed));
            verify_current_forest("small_common_cluster_repair_pass");
            publish_if_better();
            return saved;
        }
        if (best_saved > 0 && !best_seed.empty()) {
            install_components(std::move(best_seed));
            verify_current_forest("small_common_cluster_repair_pass_tail");
            publish_if_better();
            return best_saved;
        }
        return 0;
    }

    // Dedicated exact-core repair (DEV): mirrors research/solver_v2.py. Enumerate
    // common clades (a clade in BOTH trees) that are CLEAN w.r.t. the current
    // forest (== a union of whole current components) and small enough to solve
    // the induced sub-instance EXACTLY (verified Whidden B&B), then splice the
    // proven-optimal sub-forest in only when it strictly reduces the component
    // count. Never regresses (optimum <= current; strict-improvement gated;
    // re-validated by install/publish). Complements kernelization.
    int exact_core_repair_pass(double max_pass_seconds = 20.0) {
        if (!use_exact_core_) return 0;
        if (deadline::expired() || deadline::seconds_left() < 1.5 || n < 50) return 0;
        if (!best_snapshot_.empty()) install_components(vector<Component>(best_snapshot_));

        // Match by LEAF-SET (a clade in both trees with the same leaves), NOT by
        // topology key -- the payoff is exactly the clades whose internal
        // topology DIFFERS between the trees (nonzero local distance).
        unordered_map<LeafSetKey, int, LeafSetKeyHash> other;
        other.reserve(T[1].child.size() * 2 + 1);
        for (int u = 0; u < (int)T[1].child.size(); ++u) {
            int sz = T[1].subtree_size[u];
            if (sz < 4 || sz > EXACT_CORE_MAX_LEAVES) continue;
            other.emplace(leaf_set_key(T[1].key[u]), u);
        }
        struct Cl { int u, v, sz; };
        vector<Cl> clades;
        for (int u = 0; u < (int)T[0].child.size(); ++u) {
            int sz = T[0].subtree_size[u];
            if (sz < 4 || sz > EXACT_CORE_MAX_LEAVES) continue;
            auto it = other.find(leaf_set_key(T[0].key[u]));
            if (it == other.end()) continue;
            clades.push_back({u, it->second, sz});
        }
        sort(clades.begin(), clades.end(), [](const Cl& a, const Cl& b){ return a.sz < b.sz; });

        int total_saved = 0;
        double start_left = deadline::seconds_left();
        // Cap how long one pass may spend so exact repair stays a bounded polish
        // and never starves the anytime search at the full 298 s budget.
        double pass_budget = max(0.5, min(start_left * 0.5, max_pass_seconds));
        vector<int> hit;
        for (const auto& cl : clades) {
            if (deadline::expired() || deadline::seconds_left() < 1.0) break;
            if (start_left - deadline::seconds_left() > pass_budget) break;
            LeafSetKey ckey = leaf_set_key(T[0].key[cl.u]);
            if (exact_capped_.count(ckey)) continue;   // hopeless clade, skip re-solve
            if ((int)hit.size() < (int)comps.size()) hit.assign(comps.size(), 0);

            vector<int> labels = T[0].subtree_labels(cl.u);
            vector<int> touched;
            for (int lab : labels) {
                int id = label_to_comp[lab];
                if (id < 0 || id >= (int)comps.size() || !comps[id].active) continue;
                if (hit[id] == 0) touched.push_back(id);
                ++hit[id];
            }
            vector<int> inside_ids; int crossing = 0;
            for (int id : touched) {
                if (hit[id] == comps[id].size) inside_ids.push_back(id);
                else ++crossing;
                hit[id] = 0;
            }
            if (crossing != 0) continue;
            int inside_parts = (int)inside_ids.size();
            if (inside_parts < 2) continue;   // already one component -> nothing to gain

            // Guard against a (astronomically rare) leaf-set hash collision:
            // require the two clades to have identical leaf sets before solving.
            if (T[1].subtree_labels(cl.v) != labels) continue;

            vector<int> parent_to_local(n, -1);
            for (int i = 0; i < (int)labels.size(); ++i) parent_to_local[labels[i]] = i;
            string nw0 = T[0].subtree_newick_remap(cl.u, parent_to_local); nw0.push_back(';');
            string nw1 = T[1].subtree_newick_remap(cl.v, parent_to_local); nw1.push_back(';');
            auto exact = exactmaf::solve(nw0, nw1, (int)labels.size(), EXACT_CORE_NODE_CAP, 0.8);
            if (!exact) { exact_capped_[ckey] = 1; continue; }   // memoize hopeless clades
            if ((int)exact->size() >= inside_parts) continue;

            vector<Component> parts; parts.reserve(exact->size());
            bool good = true;
            for (auto& comp_local : *exact) {
                vector<int> pls; pls.reserve(comp_local.size());
                for (int lab : comp_local) {
                    if (lab < 0 || lab >= (int)labels.size()) { good = false; break; }
                    pls.push_back(labels[lab]);
                }
                if (!good) break;
                auto comp = build_agreement_component_unbounded(std::move(pls));
                if (!comp) { good = false; break; }
                parts.push_back(std::move(*comp));
            }
            if (!good) continue;

            vector<unsigned char> remove(comps.size(), 0);
            bool ok = true;
            for (int id : inside_ids) {
                if (id < 0 || id >= (int)comps.size() || !comps[id].active) { ok = false; break; }
                remove[id] = 1;
            }
            if (!ok) continue;
            vector<Component> seed;
            seed.reserve(active_count() - inside_parts + (int)parts.size());
            for (const Component& c : comps) if (c.active && !remove[c.id]) seed.push_back(c);
            for (auto& c : parts) seed.push_back(std::move(c));
            if (!seed_components_compatible(seed)) continue;
            install_components(std::move(seed));
            publish_if_better();
            total_saved += inside_parts - (int)exact->size();
            hit.assign(comps.size(), 0);   // comps rebuilt by install
        }
        return total_saved;
    }

    vector<Component> common_cluster_decomposition_for(const CommonClusterCandidate& cand) {
        if (deadline::expired() || deadline::seconds_left() < 18.0) return {};
        vector<int> cluster = cand.labels;
        vector<unsigned char> in_cluster(n, 0);
        for (int lab : cluster) in_cluster[lab] = 1;
        vector<int> outside;
        outside.reserve(n - cand.size);
        for (int lab = 0; lab < n; ++lab) if (!in_cluster[lab]) outside.push_back(lab);
        if (outside.empty()) return {};

        auto make_bottom = [&](bool marker, vector<int>& local_to_parent, string& nw0, string& nw1) {
            local_to_parent = cluster;
            vector<int> parent_to_local(n, -1);
            for (int i = 0; i < (int)local_to_parent.size(); ++i) parent_to_local[local_to_parent[i]] = i;
            int marker_label = -1;
            if (marker) {
                marker_label = (int)local_to_parent.size();
                local_to_parent.push_back(-1);
            }
            nw0 = T[0].subtree_newick_remap(cand.node0, parent_to_local);
            nw1 = T[1].subtree_newick_remap(cand.node1, parent_to_local);
            if (marker) {
                string m = to_string(marker_label + 1);
                nw0 = "(" + nw0 + "," + m + ")";
                nw1 = "(" + nw1 + "," + m + ")";
            }
            nw0.push_back(';');
            nw1.push_back(';');
        };

        auto make_top = [&](bool marker, vector<int>& local_to_parent, string& nw0, string& nw1) {
            local_to_parent = outside;
            vector<int> parent_to_local(n, -1);
            for (int i = 0; i < (int)local_to_parent.size(); ++i) parent_to_local[local_to_parent[i]] = i;
            int marker_label = -1;
            if (marker) {
                marker_label = (int)local_to_parent.size();
                local_to_parent.push_back(-1);
                nw0 = top_with_cluster_marker_rec(0, T[0].root, cand.node0,
                                                  parent_to_local, marker_label);
                nw1 = top_with_cluster_marker_rec(1, T[1].root, cand.node1,
                                                  parent_to_local, marker_label);
            } else {
                nw0 = top_with_cluster_marker_rec(0, T[0].root, cand.node0,
                                                  parent_to_local, -1);
                nw1 = top_with_cluster_marker_rec(1, T[1].root, cand.node1,
                                                  parent_to_local, -1);
            }
            nw0.push_back(';');
            nw1.push_back(';');
        };

        vector<int> bno_map, tno_map, bmark_map, tmark_map;
        string bno0, bno1, tno0, tno1, bmark0, bmark1, tmark0, tmark1;
        make_bottom(false, bno_map, bno0, bno1);
        make_top(false, tno_map, tno0, tno1);
        MappedSubproblemResult bno = solve_mapped_subproblem(bno_map, bno0, bno1);
        if (!bno.ok || deadline::expired()) return {};
        MappedSubproblemResult tno = solve_mapped_subproblem(tno_map, tno0, tno1);
        if (!tno.ok || deadline::expired()) return {};

        vector<Component> closed;
        closed.reserve(bno.regular.size() + tno.regular.size());
        for (auto& c : bno.regular) closed.push_back(std::move(c));
        for (auto& c : tno.regular) closed.push_back(std::move(c));
        vector<Component> best = std::move(closed);

        if (!deadline::expired() && deadline::seconds_left() > 12.0) {
            make_bottom(true, bmark_map, bmark0, bmark1);
            make_top(true, tmark_map, tmark0, tmark1);
            MappedSubproblemResult bmark = solve_mapped_subproblem(bmark_map, bmark0, bmark1);
            MappedSubproblemResult tmark = solve_mapped_subproblem(tmark_map, tmark0, tmark1);
            if (bmark.ok && tmark.ok && bmark.marker_found && tmark.marker_found) {
                vector<int> bridge = bmark.marker_labels;
                bridge.insert(bridge.end(), tmark.marker_labels.begin(), tmark.marker_labels.end());
                auto bridge_comp = build_agreement_component_unbounded(std::move(bridge));
                if (bridge_comp) {
                    vector<Component> open;
                    open.reserve(bmark.regular.size() + tmark.regular.size() + 1);
                    for (auto& c : bmark.regular) open.push_back(std::move(c));
                    for (auto& c : tmark.regular) open.push_back(std::move(c));
                    open.push_back(std::move(*bridge_comp));
                    if (open.size() < best.size()) best = std::move(open);
                }
            }
        }

        if (!best.empty() && seed_components_compatible(best)) return best;
        return {};
    }

    vector<Component> common_cluster_decomposition_seed(int variant) {
        if (deadline::expired() || best_snapshot_.empty() || deadline::seconds_left() < 30.0) return {};
        install_components(vector<Component>(best_snapshot_));
        vector<CommonClusterCandidate> candidates = common_cluster_candidates(variant);
        for (const CommonClusterCandidate& cand : candidates) {
            if (deadline::expired() || deadline::seconds_left() < 24.0) break;
            int outside = n - cand.size;
            if (outside < COMMON_CLUSTER_MIN_SIZE) continue;
            if (cand.crossing == 0) continue;
            vector<Component> seed = common_cluster_decomposition_for(cand);
            if (seed.empty()) continue;
            int slack = max(120, n / 60);
            if ((int)seed.size() <= best_count_ + slack) return seed;
        }
        return {};
    }

    vector<Component> pair_portfolio_seed(int variant) {
        struct PairCand {
            int cost = 0;
            uint64_t key = 0;
            int a = -1;
            int b = -1;
            Component comp;
        };

        vector<pair<int,int>> pairs;
        pairs.reserve((size_t)n * 16 + 16);
        for (int ti = 0; ti < 2; ++ti) {
            const Tree& tr = T[ti];
            for (int u : tr.post) {
                int l = tr.child[u][0], r = tr.child[u][1];
                if (l == -1 || r == -1) continue;
                if (tr.child[l][0] == -1 && tr.child[r][0] == -1) {
                    int a = tr.leaf_label[l], b = tr.leaf_label[r];
                    if (a > b) swap(a, b);
                    pairs.push_back({a, b});
                }
            }
            for (int i = 0; i < (int)tr.leaf_order.size(); ++i) {
                for (int d = 1; d <= PAIR_SEED_ORDER_RADIUS && i + d < (int)tr.leaf_order.size(); ++d) {
                    int a = tr.leaf_order[i], b = tr.leaf_order[i + d];
                    if (a > b) swap(a, b);
                    pairs.push_back({a, b});
                }
            }
        }
        sort(pairs.begin(), pairs.end());
        pairs.erase(unique(pairs.begin(), pairs.end()), pairs.end());

        vector<PairCand> cands;
        cands.reserve(pairs.size());
        for (auto [a, b] : pairs) {
            if (deadline::expired()) break;
            auto comp = build_component_from_labels({a, b}, 2, false);
            if (!comp) continue;
            int cost = (int)comp->edges[0].size() + (int)comp->edges[1].size();
            uint64_t key = splitmix64(((uint64_t)(uint32_t)a << 32) ^ (uint32_t)b ^
                                      (0x9e3779b97f4a7c15ULL * (uint64_t)(variant + 1)));
            cands.push_back(PairCand{cost, key, a, b, std::move(*comp)});
        }
        sort(cands.begin(), cands.end(), [&](const PairCand& x, const PairCand& y) {
            if (variant == 0) {
                if (x.cost != y.cost) return x.cost < y.cost;
                return x.key < y.key;
            }
            if (variant == 1) {
                if (x.key != y.key) return x.key < y.key;
                return x.cost < y.cost;
            }
            if (variant == 2) {
                if (x.cost != y.cost) return x.cost > y.cost;
                return x.key < y.key;
            }
            uint64_t sx = x.key ^ splitmix64((uint64_t)x.cost + 0x517cc1b727220a95ULL);
            uint64_t sy = y.key ^ splitmix64((uint64_t)y.cost + 0x517cc1b727220a95ULL);
            return sx < sy;
        });

        vector<Component> seed;
        seed.reserve(n);
        vector<unsigned char> leaf_used(n, 0);
        array<vector<unsigned char>, 2> edge_used;
        edge_used[0].assign(T[0].child.size(), 0);
        edge_used[1].assign(T[1].child.size(), 0);

        for (auto& cand : cands) {
            if (deadline::expired()) break;
            if (leaf_used[cand.a] || leaf_used[cand.b]) continue;
            bool conflict = false;
            for (int ti = 0; ti < 2 && !conflict; ++ti) {
                for (int e : cand.comp.edges[ti]) {
                    if (edge_used[ti][e]) { conflict = true; break; }
                }
            }
            if (conflict) continue;
            leaf_used[cand.a] = leaf_used[cand.b] = 1;
            for (int ti = 0; ti < 2; ++ti) for (int e : cand.comp.edges[ti]) edge_used[ti][e] = 1;
            seed.push_back(std::move(cand.comp));
        }
        for (int lab = 0; lab < n; ++lab) {
            if (!leaf_used[lab]) seed.push_back(build_singleton(lab));
        }
        return seed;
    }

    struct CherryGroup {
        vector<int> labels;
        bool active = true;
        bool output = true;
    };

    vector<pair<int,int>> residual_group_cherries(
        int ti,
        const vector<int>& label_group,
        const vector<CherryGroup>& groups
    ) {
        const Tree& tr = T[ti];
        vector<int> first(tr.child.size(), -1), second(tr.child.size(), -1), cnt(tr.child.size(), 0);
        vector<uint64_t> encoded;
        encoded.reserve(n + 16);
        auto add_id = [&](int id, int& c, int& a, int& b) {
            if (id < 0 || id >= (int)groups.size() || !groups[id].active) return;
            if (c >= 1 && id == a) return;
            if (c >= 2 && id == b) return;
            if (c == 0) { a = id; c = 1; }
            else if (c == 1) { b = id; c = 2; }
            else c = 3;
        };
        for (int u : tr.post) {
            if (tr.child[u][0] == -1) {
                int lab = tr.leaf_label[u];
                int gid = label_group[lab];
                if (gid >= 0 && gid < (int)groups.size() && groups[gid].active) {
                    first[u] = gid;
                    cnt[u] = 1;
                }
            } else {
                int a = -1, b = -1, c = 0;
                for (int ch : tr.child[u]) {
                    if (cnt[ch] == 1) add_id(first[ch], c, a, b);
                    else if (cnt[ch] == 2) { add_id(first[ch], c, a, b); add_id(second[ch], c, a, b); }
                    else if (cnt[ch] > 2) c = 3;
                    if (c > 2) break;
                }
                cnt[u] = min(c, 3);
                first[u] = a;
                second[u] = b;
                if (c == 2 && a >= 0 && b >= 0 && a != b) {
                    if (a > b) swap(a, b);
                    encoded.push_back(((uint64_t)(uint32_t)a << 32) | (uint32_t)b);
                }
            }
        }
        sort(encoded.begin(), encoded.end());
        encoded.erase(unique(encoded.begin(), encoded.end()), encoded.end());
        vector<pair<int,int>> pairs;
        pairs.reserve(encoded.size());
        for (uint64_t x : encoded) pairs.push_back({(int)(x >> 32), (int)(x & 0xffffffffu)});
        return pairs;
    }

    bool seed_component_safe(
        const Component& c,
        const vector<unsigned char>& leaf_used,
        const array<vector<unsigned char>, 2>& edge_used
    ) const {
        for (int lab : c.labels) if (leaf_used[lab]) return false;
        for (int ti = 0; ti < 2; ++ti) {
            for (int e : c.edges[ti]) if (edge_used[ti][e]) return false;
        }
        return true;
    }

    void mark_seed_component(
        const Component& c,
        vector<unsigned char>& leaf_used,
        array<vector<unsigned char>, 2>& edge_used
    ) const {
        for (int lab : c.labels) leaf_used[lab] = 1;
        for (int ti = 0; ti < 2; ++ti) for (int e : c.edges[ti]) edge_used[ti][e] = 1;
    }

    void collect_active_groups_limited(
        int ti,
        int node,
        const vector<int>& label_group,
        const vector<CherryGroup>& groups,
        int forbidden_a,
        int forbidden_b,
        vector<int>& out
    ) const {
        vector<int> st = {node};
        while (!st.empty() && (int)out.size() < CHERRY_CUT_SIDE_LIMIT) {
            int u = st.back();
            st.pop_back();
            if (T[ti].child[u][0] == -1) {
                int gid = label_group[T[ti].leaf_label[u]];
                if (gid < 0 || gid == forbidden_a || gid == forbidden_b ||
                    gid >= (int)groups.size() || !groups[gid].active) {
                    continue;
                }
                if (find(out.begin(), out.end(), gid) == out.end()) out.push_back(gid);
            } else {
                st.push_back(T[ti].child[u][0]);
                st.push_back(T[ti].child[u][1]);
            }
        }
    }

    vector<int> residual_side_groups_for_pair(
        int ti,
        int ga,
        int gb,
        const vector<int>& label_group,
        const vector<CherryGroup>& groups
    ) const {
        vector<int> out;
        if (ga < 0 || gb < 0 || ga >= (int)groups.size() || gb >= (int)groups.size()) return out;
        if (groups[ga].labels.empty() || groups[gb].labels.empty()) return out;
        int ua = T[ti].label_to_node[groups[ga].labels[0]];
        int ub = T[ti].label_to_node[groups[gb].labels[0]];
        int w = T[ti].lca(ua, ub);
        auto climb = [&](int u) {
            while (u != w && (int)out.size() < CHERRY_CUT_SIDE_LIMIT) {
                int p = T[ti].parent[u];
                int sib = (T[ti].child[p][0] == u ? T[ti].child[p][1] : T[ti].child[p][0]);
                if (sib != -1) collect_active_groups_limited(ti, sib, label_group, groups, ga, gb, out);
                u = p;
            }
        };
        climb(ua);
        climb(ub);
        return out;
    }

    vector<Component> materialize_cherry_groups(vector<CherryGroup>& groups, int max_components) {
        vector<int> ids;
        ids.reserve(groups.size());
        for (int i = 0; i < (int)groups.size(); ++i) if (groups[i].output && !groups[i].labels.empty()) ids.push_back(i);
        sort(ids.begin(), ids.end(), [&](int a, int b) {
            if (groups[a].labels.size() != groups[b].labels.size())
                return groups[a].labels.size() > groups[b].labels.size();
            return groups[a].labels < groups[b].labels;
        });

        vector<Component> seed;
        seed.reserve(n);
        vector<unsigned char> leaf_used(n, 0);
        array<vector<unsigned char>, 2> edge_used;
        edge_used[0].assign(T[0].child.size(), 0);
        edge_used[1].assign(T[1].child.size(), 0);

        for (int id : ids) {
            if (deadline::expired()) break;
            if ((int)seed.size() >= max_components) return {};
            auto comp = build_component_from_labels(groups[id].labels, CHERRY_CUT_COMPONENT_LIMIT, true);
            if (!comp || !seed_component_safe(*comp, leaf_used, edge_used)) continue;
            mark_seed_component(*comp, leaf_used, edge_used);
            seed.push_back(std::move(*comp));
        }
        for (int lab = 0; lab < n; ++lab) {
            if (leaf_used[lab]) continue;
            if ((int)seed.size() >= max_components) return {};
            Component c = build_singleton(lab);
            mark_seed_component(c, leaf_used, edge_used);
            seed.push_back(std::move(c));
        }
        return seed;
    }

    vector<Component> cherry_cut_beam_seed(int variant) {
        vector<CherryGroup> groups;
        groups.reserve((size_t)n * 2 + 16);
        vector<int> label_group(n, -1);
        for (int lab = 0; lab < n; ++lab) {
            CherryGroup g;
            g.labels = {lab};
            int id = (int)groups.size();
            groups.push_back(std::move(g));
            label_group[lab] = id;
        }

        int active = (int)groups.size();
        int output_count = active;
        for (int round = 0; round < CHERRY_CUT_MAX_ROUNDS && active > 1 && !deadline::expired(); ++round) {
            auto p0 = residual_group_cherries(0, label_group, groups);
            auto p1 = residual_group_cherries(1, label_group, groups);
            if (deadline::expired()) break;
            if (p0.empty() && p1.empty()) break;

            vector<pair<int,int>> common;
            common.reserve(min(p0.size(), p1.size()));
            set_intersection(p0.begin(), p0.end(), p1.begin(), p1.end(), back_inserter(common));
            if (!common.empty()) {
                sort(common.begin(), common.end(), [&](const auto& A, const auto& B) {
                    int sa = (int)groups[A.first].labels.size() + (int)groups[A.second].labels.size();
                    int sb = (int)groups[B.first].labels.size() + (int)groups[B.second].labels.size();
                    if ((variant & 1) == 0 && sa != sb) return sa > sb;
                    if ((variant & 1) == 1 && sa != sb) return sa < sb;
                    uint64_t ka = splitmix64(((uint64_t)(uint32_t)A.first << 32) ^ (uint32_t)A.second ^
                                             (uint64_t)(variant + 1) * 0x9e3779b97f4a7c15ULL);
                    uint64_t kb = splitmix64(((uint64_t)(uint32_t)B.first << 32) ^ (uint32_t)B.second ^
                                             (uint64_t)(variant + 1) * 0x9e3779b97f4a7c15ULL);
                    return ka < kb;
                });
                int merged = 0;
                vector<unsigned char> used_group(groups.size(), 0);
                for (auto [a, b] : common) {
                    if (deadline::expired()) break;
                    if (a < 0 || b < 0 || a >= (int)groups.size() || b >= (int)groups.size()) continue;
                    if (!groups[a].active || !groups[b].active || used_group[a] || used_group[b]) continue;
                    vector<int> labels;
                    labels.reserve(groups[a].labels.size() + groups[b].labels.size());
                    labels.insert(labels.end(), groups[a].labels.begin(), groups[a].labels.end());
                    labels.insert(labels.end(), groups[b].labels.begin(), groups[b].labels.end());
                    auto comp = build_component_from_labels(labels, CHERRY_CUT_COMPONENT_LIMIT, true);
                    if (!comp) continue;
                    labels = std::move(comp->labels);
                    CherryGroup g;
                    g.labels = std::move(labels);
                    int nid = (int)groups.size();
                    groups.push_back(std::move(g));
                    used_group.push_back(0);
                    groups[a].active = false;
                    groups[b].active = false;
                    groups[a].output = false;
                    groups[b].output = false;
                    used_group[a] = used_group[b] = 1;
                    for (int lab : groups[nid].labels) label_group[lab] = nid;
                    --active;
                    --output_count;
                    ++merged;
                }
                if (merged > 0) continue;
            }

            const vector<pair<int,int>>& ref_pairs = ((round + variant) & 1) ? p1 : p0;
            const vector<pair<int,int>>& other_pairs = ((round + variant) & 1) ? p0 : p1;
            int other_tree = ((round + variant) & 1) ? 0 : 1;
            if (ref_pairs.empty()) break;
            vector<int> conflict_count(groups.size(), 0);
            int inspected = 0;
            for (auto [a, b] : ref_pairs) {
                if (binary_search(other_pairs.begin(), other_pairs.end(), pair<int,int>{a, b})) continue;
                bool scored_side = false;
                if (inspected++ < CHERRY_CUT_CONFLICT_INSPECT) {
                    auto side = residual_side_groups_for_pair(other_tree, a, b, label_group, groups);
                    if (!side.empty()) {
                        for (int gid : side) {
                            if (gid >= 0 && gid < (int)conflict_count.size()) conflict_count[gid] += 4;
                        }
                        scored_side = true;
                    }
                }
                if (!scored_side) {
                    if (a >= 0 && a < (int)groups.size() && groups[a].active) ++conflict_count[a];
                    if (b >= 0 && b < (int)groups.size() && groups[b].active) ++conflict_count[b];
                }
            }
            vector<int> cut_ids;
            for (int id = 0; id < (int)groups.size(); ++id) {
                if (id < (int)conflict_count.size() && conflict_count[id] > 0 && groups[id].active) cut_ids.push_back(id);
            }
            if (cut_ids.empty()) break;
            sort(cut_ids.begin(), cut_ids.end(), [&](int a, int b) {
                long long sa = (long long)conflict_count[a] * 1000000LL;
                long long sb = (long long)conflict_count[b] * 1000000LL;
                int za = (int)groups[a].labels.size();
                int zb = (int)groups[b].labels.size();
                if (variant % 3 == 0) { sa -= za * 4096LL; sb -= zb * 4096LL; }
                else if (variant % 3 == 1) { sa += za * 4096LL; sb += zb * 4096LL; }
                uint64_t ka = splitmix64((uint64_t)(uint32_t)a ^ ((uint64_t)(variant + 17) << 32));
                uint64_t kb = splitmix64((uint64_t)(uint32_t)b ^ ((uint64_t)(variant + 17) << 32));
                if (sa != sb) return sa > sb;
                return ka < kb;
            });
            int divisor = (variant % 3 == 1 ? 48 : (variant % 3 == 2 ? 80 : 64));
            int cut_count = max(1, active / divisor);
            cut_count = min(cut_count, (int)cut_ids.size());
            for (int i = 0; i < cut_count; ++i) {
                int id = cut_ids[i];
                if (!groups[id].active) continue;
                groups[id].active = false;
                --active;
            }
        }
        for (auto& g : groups) g.active = false;
        if (output_count >= best_count_) return {};
        return materialize_cherry_groups(groups, best_count_);
    }

    void run_seed(vector<Component> seed) {
        if (deadline::expired() || seed.empty()) return;
        int before_best = best_count_;
        install_components(std::move(seed));
        publish_if_better();
        if (deadline::expired()) return;
        improvement_passes();
        publish_if_better();
        if (!deadline::expired() && deadline::seconds_left() > 8.0) {
            int saved = deep_local_repair_pass();
            if (saved > 0) {
                publish_if_better();
                improvement_passes();
                publish_if_better();
            }
        }
        if (!deadline::expired() && best_count_ < before_best && deadline::seconds_left() > 2.0) {
            small_common_cluster_repair_pass();
        }
    }

    void improvement_passes() {
        for (int pass = 0; pass < 5; ++pass) {
            if (deadline::expired()) return;
            int pair_added = 0;
            int added = 0;
            if (n >= 3000) {
                added = singleton_pack_pass();
                if (deadline::expired()) return;
                pair_added = singleton_pair_pass();
            } else {
                pair_added = singleton_pair_pass();
                if (deadline::expired()) return;
                added = singleton_pack_pass();
            }
            if (deadline::expired()) return;
            int packed = component_pack_pass();
            if (deadline::expired()) return;
            int corridor_merged = 0;
            if (active_count() <= 12000 && deadline::seconds_left() > 2.0) {
                corridor_merged = corridor_merge_pass();
            }
            if (deadline::expired()) return;
            int component_repaired = 0;
            if (active_count() <= 12000 && deadline::seconds_left() > 3.0) {
                component_repaired = component_repair_pass();
            }
            if (deadline::expired()) return;
            int merged_total = 0;
            for (int i = 0; i < MERGE_PASSES; ++i) {
                if (deadline::expired()) return;
                int m = merge_pass();
                if (m == 0) break;
                merged_total += m;
            }
            int repaired = 0;
            // Exact-cover local repair is useful but expensive; after cheap reductions it is safer.
            // Use the wider gate only when running with a judge-scale budget.
            int local_limit = LOCAL_REPAIR_BASE_COMPONENT_LIMIT;
            if (deadline::budget_seconds() >= 60.0 && deadline::seconds_left() > 30.0) {
                local_limit = LOCAL_REPAIR_COMPONENT_LIMIT;
            }
            if (!deadline::expired() && active_count() <= local_limit) repaired = local_repair_pass();
            if (!deadline::expired() && repaired == 0 && pass >= 1 && deadline::seconds_left() > 12.0) {
                repaired = deep_local_repair_pass();
            }
            // Conflict-directed repair: resolve merges blocked by a third
            // component. Gated to large instances, where this runs during the
            // early-phase improvement_passes the instance actually reaches (the
            // restart tail is rarely reached for large n). On mid instances it
            // yields no net gain (resolving a conflict by splitting the blocker
            // costs as much as the merge saves) and steals time, so it is off.
            int conflict_repaired = 0;
            if (!deadline::expired() && deadline::seconds_left() > 4.0 && n >= 5000 &&
                active_count() <= 12000 && merged_total == 0) {
                conflict_repaired = conflict_repair_pass(
                    CONFLICT_REPAIR_MAX_LEAVES, CONFLICT_REPAIR_MAX_BLOCKERS,
                    CONFLICT_REPAIR_MAX_WINDOWS, CONFLICT_REPAIR_MAX_CANDIDATES,
                    CONFLICT_REPAIR_MAX_SEARCH_NODES);
            }
            // Publish the current forest as a SIGTERM-safe best-so-far snapshot.
            publish_if_better();
            if (pair_added == 0 && added == 0 && packed == 0 && corridor_merged == 0 && component_repaired == 0 &&
                merged_total == 0 && repaired == 0 && conflict_repaired == 0) break;
            if (active_count() == 1) break;
        }
    }

    // ---- Best-so-far tracking for SIGTERM-safe output -------------------
    // best_lines_ holds the smallest forest we have produced so far. Every
    // time the live forest beats it, we update best_lines_ AND push it into
    // best_cache (which the signal handler reads).
    int best_count_ = INT32_MAX;
    vector<string> best_lines_;
    vector<Component> best_snapshot_;
    int deep_local_attempts_ = 0;
    int small_common_cluster_attempts_ = 0;
    bool final_common_cluster_polish_ = false;

    void publish_if_better() {
        int cur = active_count();
        if (cur <= 0) return;
        if (cur >= best_count_) return;
        verify_current_forest("publish_if_better");
        // Always-on safety net: never let an infeasible forest become the
        // best-so-far (it would be flushed verbatim on SIGTERM and disqualify
        // the whole submission). If construction ever produces an invalid
        // forest, we keep the previous valid best instead.
        if (!current_forest_valid()) {
            cerr << "warning: skipped publishing an infeasible forest (k=" << cur << ")\n";
            return;
        }
        best_count_ = cur;
        best_lines_ = output_lines();
        best_snapshot_ = active_components_snapshot();
        if (publish_cache_) best_cache::install(best_lines_);
    }

    // Ensure SOMETHING is published. Used at the very start so that even if
    // SIGTERM arrives before any improvement pass finishes, we still print
    // a valid (singleton) forest rather than nothing.
    void publish_singleton_fallback() {
        vector<string> lines;
        lines.reserve(n);
        // A kernel "singleton" is a whole common subtree -> emit it expanded as
        // one component (valid: a common subtree is an agreement component).
        if (expand_nwk_.empty())
            for (int i = 1; i <= n; ++i) lines.push_back(to_string(i) + ";");
        else
            for (int i = 0; i < n; ++i) lines.push_back(expand_nwk_[i] + ";");
        if ((int)lines.size() < best_count_) {
            best_count_ = (int)lines.size();
            best_lines_ = lines;
            if (publish_cache_) best_cache::install(best_lines_);
        }
    }

    optional<Component> maximum_agreement_subtree_component() {
        if (deadline::expired() || n <= 1 || n > MAST_DP_LEAF_LIMIT) return nullopt;
        int m0 = (int)T[0].child.size();
        int m1 = (int)T[1].child.size();
        uint64_t cells = (uint64_t)m0 * (uint64_t)m1;
        if (cells > 125000000ULL) return nullopt;
        vector<uint16_t> dp;
        try {
            dp.assign((size_t)cells, 0);
        } catch (...) {
            return nullopt;
        }
        auto at = [&](int u, int v) -> uint16_t& { return dp[(size_t)u * (size_t)m1 + (size_t)v]; };
        auto get = [&](int u, int v) -> int { return (int)dp[(size_t)u * (size_t)m1 + (size_t)v]; };

        for (int u : T[0].post) {
            if (deadline::expired()) return nullopt;
            bool lu = (T[0].child[u][0] == -1);
            int labu = lu ? T[0].leaf_label[u] : -1;
            for (int v : T[1].post) {
                bool lv = (T[1].child[v][0] == -1);
                int best = 0;
                if (lu && lv) {
                    best = (labu == T[1].leaf_label[v]) ? 1 : 0;
                } else if (lu) {
                    best = T[1].is_ancestor_node(v, T[1].label_to_node[labu]) ? 1 : 0;
                } else if (lv) {
                    int labv = T[1].leaf_label[v];
                    best = T[0].is_ancestor_node(u, T[0].label_to_node[labv]) ? 1 : 0;
                } else {
                    int a = T[0].child[u][0], b = T[0].child[u][1];
                    int c = T[1].child[v][0], d = T[1].child[v][1];
                    best = max(get(a,c) + get(b,d), get(a,d) + get(b,c));
                    best = max(best, get(a, v));
                    best = max(best, get(b, v));
                    best = max(best, get(u, c));
                    best = max(best, get(u, d));
                }
                if (best > 65535) best = 65535;
                at(u, v) = (uint16_t)best;
            }
        }
        int best_size = get(T[0].root, T[1].root);
        if (best_size < MAST_MIN_GAIN) return nullopt;

        vector<int> labels;
        labels.reserve(best_size);
        function<void(int,int)> rec = [&](int u, int v) {
            int want = get(u, v);
            if (want <= 0) return;
            bool lu = (T[0].child[u][0] == -1);
            bool lv = (T[1].child[v][0] == -1);
            if (lu && lv) {
                if (T[0].leaf_label[u] == T[1].leaf_label[v]) labels.push_back(T[0].leaf_label[u]);
                return;
            }
            if (lu) {
                int lab = T[0].leaf_label[u];
                if (T[1].is_ancestor_node(v, T[1].label_to_node[lab])) labels.push_back(lab);
                return;
            }
            if (lv) {
                int lab = T[1].leaf_label[v];
                if (T[0].is_ancestor_node(u, T[0].label_to_node[lab])) labels.push_back(lab);
                return;
            }
            int a = T[0].child[u][0], b = T[0].child[u][1];
            int c = T[1].child[v][0], d = T[1].child[v][1];
            int pair1 = get(a,c) + get(b,d);
            int pair2 = get(a,d) + get(b,c);
            if (pair1 == want && pair1 > 0) { rec(a,c); rec(b,d); return; }
            if (pair2 == want && pair2 > 0) { rec(a,d); rec(b,c); return; }
            if (get(a, v) == want) { rec(a, v); return; }
            if (get(b, v) == want) { rec(b, v); return; }
            if (get(u, c) == want) { rec(u, c); return; }
            if (get(u, d) == want) { rec(u, d); return; }
        };
        rec(T[0].root, T[1].root);
        sort(labels.begin(), labels.end());
        labels.erase(unique(labels.begin(), labels.end()), labels.end());
        if ((int)labels.size() < MAST_MIN_GAIN) return nullopt;
        auto comp = build_component_from_labels(labels, max(SMALL_RESTRICT_LIMIT, (int)labels.size()), false);
        if (!comp) return nullopt;
        return comp;
    }

    vector<Component> mast_singleton_seed() {
        vector<Component> seed;
        auto mast = maximum_agreement_subtree_component();
        if (!mast) return seed;
        vector<unsigned char> in(n, 0);
        for (int lab : mast->labels) in[lab] = 1;
        seed.reserve(1 + n - mast->size);
        seed.push_back(std::move(*mast));
        for (int lab = 0; lab < n; ++lab) if (!in[lab]) seed.push_back(build_singleton(lab));
        return seed;
    }

    vector<int> greedy_agreement_labels_rec(vector<int> labels, int u0, int u1, int variant) {
        if (deadline::expired()) return {};
        if (labels.size() <= 1) return labels;
        if (T[0].child[u0][0] == -1 || T[1].child[u1][0] == -1) {
            return labels.size() == 1 ? labels : vector<int>{};
        }

        int a = T[0].child[u0][0], b = T[0].child[u0][1];
        int c = T[1].child[u1][0], d = T[1].child[u1][1];
        array<vector<int>, 4> part;
        for (int lab : labels) {
            int s0 = (T[0].child_below(u0, T[0].label_to_node[lab]) == b);
            int s1 = (T[1].child_below(u1, T[1].label_to_node[lab]) == d);
            part[s0 * 2 + s1].push_back(lab);
        }

        int score_same = (int)part[0].size() + (int)part[3].size();
        int score_cross = (int)part[1].size() + (int)part[2].size();
        bool take_same = score_same >= score_cross;
        if (variant == 1 && score_same == score_cross) take_same = false;
        if (variant >= 2) {
            int penalty_same = abs((int)part[0].size() - (int)part[3].size());
            int penalty_cross = abs((int)part[1].size() - (int)part[2].size());
            int adjusted_same = score_same * 8 - penalty_same;
            int adjusted_cross = score_cross * 8 - penalty_cross;
            take_same = adjusted_same >= adjusted_cross;
            if (variant >= 3 && adjusted_same == adjusted_cross) {
                uint64_t h = splitmix64((uint64_t)labels.size() ^
                                        ((uint64_t)(uint32_t)labels[0] << 32) ^
                                        (uint64_t)(uint32_t)variant);
                take_same = (h & 1) == 0;
            }
        }

        vector<int> out;
        if (take_same) {
            auto x = greedy_agreement_labels_rec(std::move(part[0]), a, c, variant);
            auto y = greedy_agreement_labels_rec(std::move(part[3]), b, d, variant);
            out.reserve(x.size() + y.size());
            out.insert(out.end(), x.begin(), x.end());
            out.insert(out.end(), y.begin(), y.end());
        } else {
            auto x = greedy_agreement_labels_rec(std::move(part[1]), a, d, variant);
            auto y = greedy_agreement_labels_rec(std::move(part[2]), b, c, variant);
            out.reserve(x.size() + y.size());
            out.insert(out.end(), x.begin(), x.end());
            out.insert(out.end(), y.begin(), y.end());
        }
        sort(out.begin(), out.end());
        return out;
    }

    vector<Component> greedy_mast_singleton_seed(int variant) {
        vector<int> labels(n);
        iota(labels.begin(), labels.end(), 0);
        vector<int> mast_labels = greedy_agreement_labels_rec(std::move(labels), T[0].root, T[1].root, variant);
        sort(mast_labels.begin(), mast_labels.end());
        mast_labels.erase(unique(mast_labels.begin(), mast_labels.end()), mast_labels.end());
        if ((int)mast_labels.size() < max(16, MAST_MIN_GAIN)) return {};
        auto mast = build_component_from_labels(mast_labels, (int)mast_labels.size(), false);
        if (!mast) return {};

        vector<Component> seed;
        vector<unsigned char> in(n, 0);
        for (int lab : mast->labels) in[lab] = 1;
        seed.reserve(1 + n - mast->size);
        seed.push_back(std::move(*mast));
        for (int lab = 0; lab < n; ++lab) if (!in[lab]) seed.push_back(build_singleton(lab));
        return seed;
    }

    void greedy_partition_collect(vector<int> labels, int variant, vector<Component>& out) {
        if (deadline::expired() || labels.empty()) return;
        sort(labels.begin(), labels.end());
        labels.erase(unique(labels.begin(), labels.end()), labels.end());
        if (labels.empty()) return;
        if ((int)labels.size() == 1) {
            out.push_back(build_singleton(labels[0]));
            return;
        }

        auto accepted = build_component_from_labels(labels, GREEDY_PARTITION_VERIFY_LIMIT, true);
        if (accepted) {
            out.push_back(std::move(*accepted));
            return;
        }

        int u0 = T[0].lca_labels(labels);
        int u1 = T[1].lca_labels(labels);
        if (T[0].child[u0][0] == -1 || T[1].child[u1][0] == -1) {
            for (int lab : labels) out.push_back(build_singleton(lab));
            return;
        }

        int a = T[0].child[u0][0], b = T[0].child[u0][1];
        int c = T[1].child[u1][0], d = T[1].child[u1][1];
        array<vector<int>, 4> part;
        for (int lab : labels) {
            int s0 = (T[0].child_below(u0, T[0].label_to_node[lab]) == b);
            int s1 = (T[1].child_below(u1, T[1].label_to_node[lab]) == d);
            part[s0 * 2 + s1].push_back(lab);
        }

        int score_same = (int)part[0].size() + (int)part[3].size();
        int score_cross = (int)part[1].size() + (int)part[2].size();
        bool take_same = score_same >= score_cross;
        if (variant == 1 && score_same == score_cross) take_same = false;
        if (variant >= 2) {
            int penalty_same = abs((int)part[0].size() - (int)part[3].size());
            int penalty_cross = abs((int)part[1].size() - (int)part[2].size());
            int adjusted_same = score_same * 8 - penalty_same;
            int adjusted_cross = score_cross * 8 - penalty_cross;
            take_same = adjusted_same >= adjusted_cross;
        }

        int selected = take_same ? (int)part[0].size() + (int)part[3].size()
                                 : (int)part[1].size() + (int)part[2].size();
        if (selected <= 0 || selected == (int)labels.size()) {
            bool progressed = false;
            for (auto& p : part) {
                if (!p.empty() && (int)p.size() < (int)labels.size()) {
                    greedy_partition_collect(std::move(p), variant, out);
                    progressed = true;
                }
            }
            if (!progressed) {
                for (int lab : labels) out.push_back(build_singleton(lab));
            }
            return;
        }

        if (take_same) {
            if (!part[0].empty()) greedy_partition_collect(std::move(part[0]), variant, out);
            if (!part[3].empty()) greedy_partition_collect(std::move(part[3]), variant, out);
            if (!part[1].empty()) greedy_partition_collect(std::move(part[1]), variant, out);
            if (!part[2].empty()) greedy_partition_collect(std::move(part[2]), variant, out);
        } else {
            if (!part[1].empty()) greedy_partition_collect(std::move(part[1]), variant, out);
            if (!part[2].empty()) greedy_partition_collect(std::move(part[2]), variant, out);
            if (!part[0].empty()) greedy_partition_collect(std::move(part[0]), variant, out);
            if (!part[3].empty()) greedy_partition_collect(std::move(part[3]), variant, out);
        }
    }

    vector<Component> greedy_partition_seed(int variant) {
        if (deadline::expired() || n <= 1) return {};
        vector<int> labels(n);
        iota(labels.begin(), labels.end(), 0);
        vector<Component> seed;
        seed.reserve(n);
        greedy_partition_collect(std::move(labels), variant, seed);
        if (deadline::expired() || seed.empty()) return {};

        vector<unsigned char> seen(n, 0);
        int covered = 0;
        for (const auto& c : seed) {
            for (int lab : c.labels) {
                if (lab < 0 || lab >= n || seen[lab]) return {};
                seen[lab] = 1;
                ++covered;
            }
        }
        for (int lab = 0; lab < n; ++lab) {
            if (!seen[lab]) {
                seed.push_back(build_singleton(lab));
                ++covered;
            }
        }
        if (covered != n) return {};
        if ((int)seed.size() > n - GREEDY_PARTITION_MIN_GAIN) return {};

        array<vector<unsigned char>, 2> edge_seen;
        edge_seen[0].assign(T[0].child.size(), 0);
        edge_seen[1].assign(T[1].child.size(), 0);
        for (const auto& c : seed) {
            for (int ti = 0; ti < 2; ++ti) {
                for (int e : c.edges[ti]) {
                    if (e < 0 || e >= (int)edge_seen[ti].size()) return {};
                    if (edge_seen[ti][e]) return {};
                    edge_seen[ti][e] = 1;
                }
            }
        }
        return seed;
    }

    vector<Component> mast_packing_seed() {
        if (deadline::expired() || n <= 1 || n > MAST_DP_LEAF_LIMIT) return {};
        int m0 = (int)T[0].child.size();
        int m1 = (int)T[1].child.size();
        uint64_t cells = (uint64_t)m0 * (uint64_t)m1;
        if (cells > 125000000ULL) return {};
        vector<uint16_t> dp;
        try {
            dp.assign((size_t)cells, 0);
        } catch (...) {
            return {};
        }
        auto at = [&](int u, int v) -> uint16_t& { return dp[(size_t)u * (size_t)m1 + (size_t)v]; };
        auto get = [&](int u, int v) -> int { return (int)dp[(size_t)u * (size_t)m1 + (size_t)v]; };

        struct CellCand {
            int score;
            int u;
            int v;
            uint64_t key;
        };
        vector<CellCand> top_cells;
        top_cells.reserve(MAST_TOP_CELLS * 2);
        auto offer_cell = [&](int score, int u, int v) {
            if (score < MAST_MIN_GAIN) return;
            uint64_t key = splitmix64(((uint64_t)(uint32_t)u << 32) ^ (uint32_t)v ^
                                      ((uint64_t)(uint32_t)score * 0x9e3779b97f4a7c15ULL));
            top_cells.push_back(CellCand{score, u, v, key});
            if ((int)top_cells.size() > MAST_TOP_CELLS * 4) {
                nth_element(top_cells.begin(), top_cells.begin() + MAST_TOP_CELLS, top_cells.end(),
                            [](const CellCand& a, const CellCand& b) {
                                if (a.score != b.score) return a.score > b.score;
                                return a.key < b.key;
                            });
                top_cells.resize(MAST_TOP_CELLS);
            }
        };

        for (int u : T[0].post) {
            if (deadline::expired()) return {};
            bool lu = (T[0].child[u][0] == -1);
            int labu = lu ? T[0].leaf_label[u] : -1;
            for (int v : T[1].post) {
                bool lv = (T[1].child[v][0] == -1);
                int best = 0;
                if (lu && lv) {
                    best = (labu == T[1].leaf_label[v]) ? 1 : 0;
                } else if (lu) {
                    best = T[1].is_ancestor_node(v, T[1].label_to_node[labu]) ? 1 : 0;
                } else if (lv) {
                    int labv = T[1].leaf_label[v];
                    best = T[0].is_ancestor_node(u, T[0].label_to_node[labv]) ? 1 : 0;
                } else {
                    int a = T[0].child[u][0], b = T[0].child[u][1];
                    int c = T[1].child[v][0], d = T[1].child[v][1];
                    best = max(get(a, c) + get(b, d), get(a, d) + get(b, c));
                    best = max(best, get(a, v));
                    best = max(best, get(b, v));
                    best = max(best, get(u, c));
                    best = max(best, get(u, d));
                }
                if (best > 65535) best = 65535;
                at(u, v) = (uint16_t)best;
                offer_cell(best, u, v);
            }
        }
        offer_cell(get(T[0].root, T[1].root), T[0].root, T[1].root);
        if (top_cells.empty()) return {};
        sort(top_cells.begin(), top_cells.end(), [](const CellCand& a, const CellCand& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.key < b.key;
        });
        if ((int)top_cells.size() > MAST_TOP_CELLS) top_cells.resize(MAST_TOP_CELLS);

        auto extract_labels = [&](int start_u, int start_v) {
            vector<int> labels;
            labels.reserve(get(start_u, start_v));
            function<void(int, int)> rec = [&](int u, int v) {
                int want = get(u, v);
                if (want <= 0) return;
                bool lu = (T[0].child[u][0] == -1);
                bool lv = (T[1].child[v][0] == -1);
                if (lu && lv) {
                    if (T[0].leaf_label[u] == T[1].leaf_label[v]) labels.push_back(T[0].leaf_label[u]);
                    return;
                }
                if (lu) {
                    int lab = T[0].leaf_label[u];
                    if (T[1].is_ancestor_node(v, T[1].label_to_node[lab])) labels.push_back(lab);
                    return;
                }
                if (lv) {
                    int lab = T[1].leaf_label[v];
                    if (T[0].is_ancestor_node(u, T[0].label_to_node[lab])) labels.push_back(lab);
                    return;
                }
                int a = T[0].child[u][0], b = T[0].child[u][1];
                int c = T[1].child[v][0], d = T[1].child[v][1];
                int pair1 = get(a, c) + get(b, d);
                int pair2 = get(a, d) + get(b, c);
                if (pair1 == want && pair1 > 0) { rec(a, c); rec(b, d); return; }
                if (pair2 == want && pair2 > 0) { rec(a, d); rec(b, c); return; }
                if (get(a, v) == want) { rec(a, v); return; }
                if (get(b, v) == want) { rec(b, v); return; }
                if (get(u, c) == want) { rec(u, c); return; }
                if (get(u, d) == want) { rec(u, d); return; }
            };
            rec(start_u, start_v);
            sort(labels.begin(), labels.end());
            labels.erase(unique(labels.begin(), labels.end()), labels.end());
            return labels;
        };

        struct PackCand {
            int saved;
            int cost;
            uint64_t key;
            vector<int> labels;
            Component comp;
        };
        vector<PackCand> cands;
        cands.reserve(min(MAST_MAX_COMPONENT_CANDIDATES, (int)top_cells.size()));
        unordered_set<vector<int>, VecHash> seen;
        seen.reserve((size_t)MAST_MAX_COMPONENT_CANDIDATES * 2 + 1);
        for (const auto& cell : top_cells) {
            if (deadline::expired()) return {};
            vector<int> labels = extract_labels(cell.u, cell.v);
            if ((int)labels.size() < MAST_MIN_GAIN) continue;
            if (!seen.insert(labels).second) continue;
            auto comp = build_component_from_labels(labels, max(SMALL_RESTRICT_LIMIT, (int)labels.size()), false);
            if (!comp) continue;
            int cost = (int)comp->edges[0].size() + (int)comp->edges[1].size();
            cands.push_back(PackCand{(int)labels.size() - 1, cost, cell.key, std::move(labels), std::move(*comp)});
            if ((int)cands.size() >= MAST_MAX_COMPONENT_CANDIDATES) break;
        }
        if (cands.empty()) return {};
        sort(cands.begin(), cands.end(), [](const PackCand& a, const PackCand& b) {
            if (a.saved != b.saved) return a.saved > b.saved;
            if (a.cost != b.cost) return a.cost < b.cost;
            return a.key < b.key;
        });

        auto can_take = [&](int idx,
                            const vector<unsigned char>& leaf_used,
                            const array<vector<unsigned char>, 2>& edge_used) -> bool {
            const PackCand& cand = cands[idx];
            for (int lab : cand.labels) if (leaf_used[lab]) return false;
            for (int ti = 0; ti < 2; ++ti) {
                for (int e : cand.comp.edges[ti]) if (edge_used[ti][e]) return false;
            }
            return true;
        };
        auto mark_take = [&](int idx,
                             vector<unsigned char>& leaf_used,
                             array<vector<unsigned char>, 2>& edge_used,
                             bool val) {
            const PackCand& cand = cands[idx];
            for (int lab : cand.labels) leaf_used[lab] = (unsigned char)val;
            for (int ti = 0; ti < 2; ++ti) {
                for (int e : cand.comp.edges[ti]) edge_used[ti][e] = (unsigned char)val;
            }
        };
        auto extend_greedily = [&](const vector<int>& forced) {
            vector<int> plan;
            vector<unsigned char> forced_idx(cands.size(), 0);
            vector<unsigned char> leaf_used(n, 0);
            array<vector<unsigned char>, 2> edge_used;
            edge_used[0].assign(T[0].child.size(), 0);
            edge_used[1].assign(T[1].child.size(), 0);
            int score = 0;
            for (int idx : forced) {
                if (idx < 0 || idx >= (int)cands.size()) continue;
                if (forced_idx[idx]) continue;
                if (!can_take(idx, leaf_used, edge_used)) continue;
                forced_idx[idx] = 1;
                mark_take(idx, leaf_used, edge_used, true);
                score += cands[idx].saved;
                plan.push_back(idx);
            }
            for (int idx = 0; idx < (int)cands.size(); ++idx) {
                if (deadline::expired()) break;
                if (forced_idx[idx]) continue;
                if (!can_take(idx, leaf_used, edge_used)) continue;
                mark_take(idx, leaf_used, edge_used, true);
                score += cands[idx].saved;
                plan.push_back(idx);
            }
            return pair<int, vector<int>>(score, std::move(plan));
        };

        auto best_plan = extend_greedily({});
        if (!deadline::expired() && deadline::seconds_left() > 2.0 && !cands.empty()) {
            int pool = min((int)cands.size(), MAST_PACK_BB_MAX_CANDIDATES);
            vector<int> suffix_saved(pool + 1, 0);
            for (int i = pool - 1; i >= 0; --i) suffix_saved[i] = suffix_saved[i + 1] + cands[i].saved;

            vector<unsigned char> bb_leaf_used(n, 0);
            array<vector<unsigned char>, 2> bb_edge_used;
            bb_edge_used[0].assign(T[0].child.size(), 0);
            bb_edge_used[1].assign(T[1].child.size(), 0);
            vector<int> cur_branch, best_branch;
            int cur_score = 0, best_branch_score = 0, search_nodes = 0;

            for (int idx = 0; idx < pool; ++idx) {
                if (!can_take(idx, bb_leaf_used, bb_edge_used)) continue;
                mark_take(idx, bb_leaf_used, bb_edge_used, true);
                best_branch_score += cands[idx].saved;
                best_branch.push_back(idx);
            }
            fill(bb_leaf_used.begin(), bb_leaf_used.end(), 0);
            fill(bb_edge_used[0].begin(), bb_edge_used[0].end(), 0);
            fill(bb_edge_used[1].begin(), bb_edge_used[1].end(), 0);

            function<void(int)> dfs = [&](int idx) {
                if (deadline::expired()) return;
                if (++search_nodes > MAST_PACK_BB_MAX_NODES) return;
                if (cur_score + suffix_saved[idx] <= best_branch_score) return;
                if (idx >= pool) {
                    if (cur_score > best_branch_score) {
                        best_branch_score = cur_score;
                        best_branch = cur_branch;
                    }
                    return;
                }
                if (can_take(idx, bb_leaf_used, bb_edge_used)) {
                    mark_take(idx, bb_leaf_used, bb_edge_used, true);
                    cur_branch.push_back(idx);
                    cur_score += cands[idx].saved;
                    dfs(idx + 1);
                    cur_score -= cands[idx].saved;
                    cur_branch.pop_back();
                    mark_take(idx, bb_leaf_used, bb_edge_used, false);
                }
                dfs(idx + 1);
            };
            dfs(0);

            auto bb_plan = extend_greedily(best_branch);
            if (bb_plan.first > best_plan.first) best_plan = std::move(bb_plan);
        }

        if (best_plan.second.empty()) return {};
        vector<Component> seed;
        seed.reserve(n);
        vector<unsigned char> leaf_used(n, 0);
        for (int idx : best_plan.second) {
            if (deadline::expired()) break;
            for (int lab : cands[idx].labels) leaf_used[lab] = 1;
            seed.push_back(std::move(cands[idx].comp));
        }
        if (seed.empty()) return {};
        for (int lab = 0; lab < n; ++lab) if (!leaf_used[lab]) seed.push_back(build_singleton(lab));
        return seed;
    }

    bool solve_exact_small() {
        if (deadline::expired() || n > EXACT_SMALL_N) return false;
        int full = 1 << n;
        struct ECand { int mask; int pop; int cost; Component comp; };
        vector<ECand> cands;
        cands.reserve(full);
        vector<vector<int>> subset_labels(full);
        for (int mask = 1; mask < full; ++mask) {
            if (deadline::expired()) return false;
            int bit = mask & -mask;
            int idx = __builtin_ctz((unsigned)bit);
            subset_labels[mask] = subset_labels[mask ^ bit];
            subset_labels[mask].push_back(idx);
            auto comp = build_component_from_labels(subset_labels[mask], n, true);
            if (!comp) continue;
            int cost = (int)comp->edges[0].size() + (int)comp->edges[1].size();
            cands.push_back({mask, __builtin_popcount((unsigned)mask), cost, std::move(*comp)});
        }
        if (cands.empty()) return false;

        vector<vector<int>> by_leaf(n);
        for (int i = 0; i < (int)cands.size(); ++i) {
            int m = cands[i].mask;
            while (m) {
                int bit = m & -m;
                by_leaf[__builtin_ctz((unsigned)bit)].push_back(i);
                m ^= bit;
            }
        }
        for (auto& v : by_leaf) {
            sort(v.begin(), v.end(), [&](int a, int b) {
                if (cands[a].pop != cands[b].pop) return cands[a].pop > cands[b].pop;
                if (cands[a].cost != cands[b].cost) return cands[a].cost < cands[b].cost;
                return cands[a].mask < cands[b].mask;
            });
        }

        int best_count = n;
        vector<int> best;
        vector<int> chosen;
        int max_pop = 1;
        for (auto& c : cands) max_pop = max(max_pop, c.pop);
        array<vector<unsigned char>,2> local_used;
        local_used[0].assign(T[0].child.size(), 0);
        local_used[1].assign(T[1].child.size(), 0);
        int search_nodes = 0;
        const int SEARCH_LIMIT = 2000000;

        function<void(int)> dfs = [&](int uncovered) {
            if (deadline::expired()) return;
            if (++search_nodes > SEARCH_LIMIT) return;
            if ((int)chosen.size() >= best_count) return;
            if (uncovered == 0) {
                best = chosen;
                best_count = (int)chosen.size();
                return;
            }
            int lb = (__builtin_popcount((unsigned)uncovered) + max_pop - 1) / max_pop;
            if ((int)chosen.size() + lb >= best_count) return;

            int selected_leaf = -1;
            vector<int> options;
            int rem = uncovered;
            while (rem) {
                int bit = rem & -rem;
                int li = __builtin_ctz((unsigned)bit);
                vector<int> opts;
                for (int ci : by_leaf[li]) {
                    const ECand& cand = cands[ci];
                    if ((cand.mask & uncovered) != cand.mask) continue;
                    bool conflict = false;
                    for (int ti = 0; ti < 2 && !conflict; ++ti) {
                        for (int e : cand.comp.edges[ti]) if (local_used[ti][e]) { conflict = true; break; }
                    }
                    if (!conflict) opts.push_back(ci);
                }
                if (opts.empty()) return;
                if (selected_leaf < 0 || opts.size() < options.size()) {
                    selected_leaf = li;
                    options = std::move(opts);
                }
                rem ^= bit;
            }

            for (int ci : options) {
                ECand& cand = cands[ci];
                for (int ti = 0; ti < 2; ++ti) for (int e : cand.comp.edges[ti]) local_used[ti][e] = 1;
                chosen.push_back(ci);
                dfs(uncovered ^ cand.mask);
                chosen.pop_back();
                for (int ti = 0; ti < 2; ++ti) for (int e : cand.comp.edges[ti]) local_used[ti][e] = 0;
                if (search_nodes > SEARCH_LIMIT || best_count == 1) return;
            }
        };
        dfs(full - 1);
        if (best.empty()) return false;
        vector<Component> exact;
        exact.reserve(best.size());
        for (int ci : best) exact.push_back(std::move(cands[ci].comp));
        install_components(std::move(exact));
        return true;
    }

    void solve() {
        // Always publish a singleton fallback FIRST so that any SIGTERM-on-startup
        // still produces a valid (worst-case k=n) forest.
        publish_singleton_fallback();

        if (solve_exact_small()) { publish_if_better(); return; }
        if (deadline::expired()) return;

        vector<Component> s0 = common_clade_sweep(0);
        if (deadline::expired()) return;
        vector<Component> s1 = common_clade_sweep(1);
        if (deadline::expired()) return;
        vector<Component> best = (s1.size() < s0.size() ? std::move(s1) : std::move(s0));
        install_components(std::move(best));
        publish_if_better();
        improvement_passes();
        publish_if_better();

        // NOTE: exact-core repair runs ONLY as a final warm-start polish (near the
        // end of solve(), on the converged best). An early pass here was found to
        // steal budget from the productive seed/restart phase and regress
        // mid/large instances at short budgets, so it was removed.

        if (!deadline::expired() && deadline::budget_seconds() >= 30.0 && deadline::seconds_left() > 2.0) {
            for (int i = 0; i < 2 && !deadline::expired(); ++i) {
                int saved = small_common_cluster_repair_pass();
                if (saved <= 0) break;
                publish_if_better();
                if (deadline::seconds_left() < 2.0) break;
            }
        }
        if (!deadline::expired() && deadline::budget_seconds() >= 100.0 && deadline::seconds_left() > 20.0) {
            final_common_cluster_polish_ = true;
            for (int i = 0; i < 3 && !deadline::expired(); ++i) {
                if (deadline::seconds_left() < 12.0) break;
                int saved = small_common_cluster_repair_pass();
                if (saved <= 0) break;
                publish_if_better();
            }
            final_common_cluster_polish_ = false;
        }

        int common_seed_limit = 0;
        if (n >= 1000 && n <= 3000 && deadline::budget_seconds() >= 60.0 && deadline::seconds_left() > 35.0) {
            common_seed_limit = 1;
        }
        for (int variant = 0; variant < common_seed_limit && !deadline::expired(); ++variant) {
            if (deadline::seconds_left() < 20.0) break;
            vector<Component> seed = common_cluster_decomposition_seed(variant);
            if (!seed.empty()) run_seed(std::move(seed));
        }

        if (!deadline::expired() && deadline::seconds_left() > 1.0) {
            vector<Component> mast_seed = mast_packing_seed();
            if (mast_seed.empty()) mast_seed = mast_singleton_seed();
            if (mast_seed.empty() && deadline::budget_seconds() <= 60.0) {
                mast_seed = greedy_mast_singleton_seed(0);
            }
            // Try the MAST-derived seed even when its initial component count
            // is not better. It often gives the merge/packing passes a
            // different basin of attraction.
            run_seed(std::move(mast_seed));
        }

        int greedy_mast_variants = 1;
        for (int variant = 1; variant < greedy_mast_variants && !deadline::expired(); ++variant) {
            if (deadline::seconds_left() < 8.0) break;
            vector<Component> seed = greedy_mast_singleton_seed(variant);
            if (!seed.empty()) run_seed(std::move(seed));
        }

        if (n > MAST_DP_LEAF_LIMIT && !deadline::expired() && deadline::seconds_left() > 5.0) {
            for (int variant = 0; variant < GREEDY_PARTITION_VARIANTS && !deadline::expired(); ++variant) {
                if (deadline::seconds_left() < 5.0) break;
                vector<Component> seed = greedy_partition_seed(variant);
                if (!seed.empty() && (int)seed.size() < best_count_) {
                    run_seed(std::move(seed));
                }
            }
        }

        // Experimental giant constructor: the cherry-cut beam is a
        // conflict/heat-style constructor. SAFE only ran it for n<=3000; for
        // giants, try a tiny budgeted dose before the pair seeds. This is kept
        // in the giant_heat variant until A/B proves it is worth promoting.
        int cherry_seed_limit = (n <= 3000 ? 4 : (n >= 7000 ? 2 : 1));
        for (int variant = 0; variant < cherry_seed_limit && !deadline::expired(); ++variant) {
            if (deadline::seconds_left() < (n >= 7000 ? 8.0 : 3.0)) break;
            vector<Component> cherry_seed = cherry_cut_beam_seed(variant);
            if (!cherry_seed.empty() && (int)cherry_seed.size() < best_count_) {
                run_seed(std::move(cherry_seed));
            }
        }

        int pair_seed_limit = (n <= 3000 ? PORTFOLIO_SEEDS : (n <= 6000 ? 6 : 2));
        for (int variant = 0; variant < pair_seed_limit && !deadline::expired(); ++variant) {
            if (deadline::seconds_left() < 1.0) break;
            run_seed(pair_portfolio_seed(variant));
        }

        // Restart loop: spend the remaining budget as an anytime search. Each
        // restart perturbs the best snapshot, shatters a diverse subset of
        // components back into singletons, and lets the improvement stack try
        // to re-glue the region in a different order. We keep going even after
        // stale iterations because Optil rewards final quality and the signal
        // handler can always emit the best forest seen so far.
        if (best_snapshot_.empty()) {
            // Defensive: should be set by publish_if_better, but fall back to
            // whatever is currently installed.
            best_snapshot_ = active_components_snapshot();
        }
        double final_repair_reserve = 0.0;
        if (n >= 1000 && deadline::budget_seconds() >= 100.0) {
            final_repair_reserve = deadline::budget_seconds() >= 180.0 ? 32.0 : 14.0;
        }
        uint32_t seed_val = 0xC0FFEEu ^ (uint32_t)(orig_n_ > 0 ? orig_n_ : n);
        if (const char* s = std::getenv("PACE_SEED")) seed_val = (uint32_t)strtoul(s, nullptr, 10);
        std::mt19937 rng(seed_val);
        int restart = 0;
        double final_repair_margin = deadline::budget_seconds() >= 180.0 ? 60.0 : 8.0;
        while (!deadline::expired()) {
            if (deadline::seconds_left() < 1.0) break;
            if (final_repair_reserve > 0.0 &&
                deadline::seconds_left() < final_repair_reserve + final_repair_margin) break;
            if (best_count_ <= 1) break;
            ++restart;
            if (!perturb_and_rerun(rng, restart)) break;
        }

        // Re-install the best snapshot so output_lines() at the end matches.
        install_components(vector<Component>(best_snapshot_));
        if (final_repair_reserve > 0.0 && !deadline::expired() && deadline::seconds_left() > 2.0) {
            small_common_cluster_attempts_ = 0;
            final_common_cluster_polish_ = true;
            int final_passes = deadline::budget_seconds() >= 180.0 ? 8 : 4;
            for (int i = 0; i < final_passes && !deadline::expired(); ++i) {
                if (deadline::seconds_left() < 2.0) break;
                int saved = small_common_cluster_repair_pass();
                if (saved <= 0) break;
                publish_if_better();
            }
            final_common_cluster_polish_ = false;
            if (!best_snapshot_.empty()) install_components(vector<Component>(best_snapshot_));
        }
        // Final exact-core repair on the converged forest (warm-start, like
        // research/solver_v2.py). Never regresses.
        if (use_exact_core_ && !deadline::expired() && deadline::seconds_left() > 1.5) {
            for (int i = 0; i < 4 && !deadline::expired(); ++i) {
                int saved = exact_core_repair_pass(45.0);
                publish_if_better();
                if (saved <= 0 || deadline::seconds_left() < 1.5) break;
            }
            if (!best_snapshot_.empty()) install_components(vector<Component>(best_snapshot_));
        }
        publish_if_better();
    }

    // Pick a random subset of the current best snapshot to shatter, install
    // the shattered forest, and run a single improvement pass round. Keeps
    // the result if it beats best_count_; otherwise restores nothing
    // (best_snapshot_ is left as-is and the caller will reinstall it).
    bool perturb_and_rerun(std::mt19937& rng, int restart_idx) {
        if (best_snapshot_.empty()) return false;
        // Reinstall a fresh restart basis. Most restarts use the current global
        // best, but periodic restarts revisit earlier strong basins, which
        // prevents a good seed from monopolizing the anytime search.
        const vector<Component>* base = &best_snapshot_;
        install_components(vector<Component>(*base));

        // Compute shatter ratio: alternate between gentle (5%) and aggressive
        // (up to ~25%) perturbations.
        int active = active_count();
        if (active <= 2) return false;
        double ratio;
        switch (restart_idx % 8) {
            case 0: ratio = 0.03; break;
            case 1: ratio = 0.07; break;
            case 2: ratio = 0.12; break;
            case 3: ratio = 0.20; break;
            case 4: ratio = 0.30; break;
            case 5: ratio = 0.15; break;
            case 6: ratio = 0.25; break;
            default: ratio = 0.10; break;
        }
        int target = std::max(2, (int)std::round(active * ratio));
        target = std::min(target, active - 1);
        // Collect active component IDs and pick `target` of them.
        vector<int> active_ids;
        active_ids.reserve(active);
        for (int i = 0; i < (int)comps.size(); ++i) {
            if (comps[i].active) active_ids.push_back(i);
        }
        std::shuffle(active_ids.begin(), active_ids.end(), rng);
        if (target > (int)active_ids.size()) target = (int)active_ids.size();
        active_ids.resize(target);

        // Collect the labels of the chosen components, then shatter them into
        // singletons in place.
        vector<int> shattered_labels;
        for (int id : active_ids) {
            for (int lab : comps[id].labels) shattered_labels.push_back(lab);
            remove_component(id);
        }
        for (int lab : shattered_labels) {
            if (label_to_comp[lab] == -1) add_component(build_singleton(lab));
        }
        if (deadline::expired()) return true;
        int before_best = best_count_;
        improvement_passes();
        publish_if_better();
        if (!deadline::expired() && best_count_ < before_best && deadline::seconds_left() > 2.0) {
            bool old_final_polish = final_common_cluster_polish_;
            int old_common_attempts = small_common_cluster_attempts_;
            small_common_cluster_attempts_ = 0;
            if (deadline::budget_seconds() >= 100.0 && deadline::seconds_left() > 4.0) {
                final_common_cluster_polish_ = true;
            }
            int polish_passes = final_common_cluster_polish_
                ? (deadline::budget_seconds() >= 180.0 ? 4 : 2)
                : 1;
            for (int i = 0; i < polish_passes && !deadline::expired(); ++i) {
                if (deadline::seconds_left() < 2.0) break;
                int saved = small_common_cluster_repair_pass();
                if (saved <= 0) break;
                publish_if_better();
            }
            final_common_cluster_polish_ = old_final_polish;
            small_common_cluster_attempts_ = old_common_attempts;
        }
        return true;
    }

    vector<string> output_lines() const {
        vector<const Component*> active;
        active.reserve(comps.size());
        for (const auto& c : comps) if (c.active) active.push_back(&c);
        sort(active.begin(), active.end(), [](const Component* a, const Component* b) {
            if (a->labels.empty() || b->labels.empty()) return a->id < b->id;
            if (a->labels[0] != b->labels[0]) return a->labels[0] < b->labels[0];
            if (a->size != b->size) return a->size < b->size;
            return a->id < b->id;
        });
        vector<string> lines;
        lines.reserve(active.size());
        if (expand_nwk_.empty()) {
            for (auto* c : active) lines.push_back(c->newick + ";");
        } else {
            for (auto* c : active)
                lines.push_back(kern::expand_component(c->newick, expand_nwk_) + ";");
        }
        return lines;
    }

    vector<string> best_output_lines() const {
        if (!best_lines_.empty()) return best_lines_;
        return output_lines();
    }
};

// ============================================================================
// Lossless common-subtree (cherry) kernelization  [DEV: heuristic_dev_kernel]
// ----------------------------------------------------------------------------
// Iteratively collapse maximal common pendant subtrees. Base case: a cherry
// (two sibling leaves) that is a cherry in BOTH trees -> replace by one leaf;
// this can expose new common cherries, so iterate. This is the classic
// Bordewich-Semple subtree reduction: it is DISTANCE / OPTIMALITY PRESERVING
// (a common pendant subtree is intra-component in every MAF), so the reduced
// instance has the same optimum and any valid forest on it expands to a valid
// forest on the original. We solve the smaller kernel (the throughput-bound
// search does more effective work) and expand components back at output time.
// ============================================================================
namespace kern {

struct KTree {
    vector<int> l, r, lab;   // per node: children (-1 leaf), leaf label (>=0) or -1
    int root = -1;
    int newick_parse(const string& s, size_t& p) {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
        if (s[p] == '(') {
            ++p;
            int a = newick_parse(s, p);
            while (p < s.size() && s[p] != ',') ++p;
            ++p; // skip ','
            int b = newick_parse(s, p);
            while (p < s.size() && s[p] != ')') ++p;
            ++p; // skip ')'
            int id = (int)l.size();
            l.push_back(a); r.push_back(b); lab.push_back(-1);
            return id;
        }
        size_t j = p;
        while (p < s.size() && isdigit((unsigned char)s[p])) ++p;
        int value = std::stoi(s.substr(j, p - j)) - 1; // store 0-based
        int id = (int)l.size();
        l.push_back(-1); r.push_back(-1); lab.push_back(value);
        return id;
    }
    void parse(const string& s) { size_t p = 0; root = newick_parse(s, p); }
    string serialize(int node, const vector<int>& relabel) const {
        if (l[node] == -1) return to_string(relabel[lab[node]] + 1);
        return "(" + serialize(l[node], relabel) + "," + serialize(r[node], relabel) + ")";
    }
};

// Returns true if a reduction happened. Fills reduced newicks (relabeled 1..K)
// and expand_nwk[k] = original-labeled newick string for kernel leaf k (0-based).
inline bool kernelize(const string& nw0, const string& nw1, int n,
                      string& out0, string& out1, vector<string>& expand_nwk) {
    KTree T0, T1;
    T0.parse(nw0);
    T1.parse(nw1);
    vector<string> enw;
    enw.reserve((size_t)n * 2 + 4);
    for (int i = 0; i < n; ++i) enw.push_back(to_string(i + 1));
    int next_label = n;

    auto build_cherries = [](const KTree& T) {
        unordered_map<long long, int> m;
        m.reserve(T.l.size());
        for (int u = 0; u < (int)T.l.size(); ++u) {
            int cl = T.l[u], cr = T.r[u];
            if (cl == -1) continue;                       // leaf
            if (T.l[cl] != -1 || T.l[cr] != -1) continue; // not a cherry
            int a = T.lab[cl], b = T.lab[cr];
            if (a > b) std::swap(a, b);
            m[(long long)a * 4000003LL + b] = u;
        }
        return m;
    };

    // Cherry reduction is a single-threaded O(nodes) scan per round, so a long
    // common "ladder" (deep caterpillar) costs O(n^2) time. Bound it to a small
    // slice of the budget: a PARTIAL reduction is still optimality-preserving,
    // and the full solver's O(n) common-clade sweep already collapses giant
    // common subtrees, so bailing early here loses nothing and hands the rest of
    // the budget to the actual search instead of burning it in the kernel.
    const double kern_start_left = deadline::seconds_left();
    const double kern_cap = std::min(20.0, std::max(1.0, 0.05 * deadline::budget_seconds()));
    bool any = false;
    while (true) {
        // Stop reducing if the wall-clock budget is gone or the kernel slice is
        // used up: a valid singleton fallback is already cached, and a PARTIAL
        // cherry reduction is still optimality-preserving, so breaking here is
        // always safe.
        if (deadline::expired() || (kern_start_left - deadline::seconds_left()) > kern_cap) break;
        auto c0 = build_cherries(T0);
        auto c1 = build_cherries(T1);
        vector<pair<long long,int>> common;
        common.reserve(c0.size());
        for (auto& kv : c0) if (c1.count(kv.first)) common.push_back(kv);
        if (common.empty()) break;
        for (auto& kv : common) {
            long long key = kv.first;
            int u0 = kv.second, u1 = c1[key];
            int a = T0.lab[T0.l[u0]], b = T0.lab[T0.r[u0]];
            if (a > b) std::swap(a, b);
            int nl = next_label++;
            if ((int)enw.size() <= nl) enw.resize(nl + 1);
            enw[nl] = "(" + enw[a] + "," + enw[b] + ")";
            // a and b are subsumed into the new super-leaf nl: they are removed
            // from both trees and can never be a reachable leaf or a cherry child
            // again, so their expansion strings are dead. Free them. Without this,
            // a long common ladder (deep caterpillar) accumulates O(K^2) total
            // characters across all intermediate super-leaves and blows the 8 GB
            // memory limit (OOM -> SIGKILL -> empty output -> disqualification).
            string().swap(enw[a]);
            string().swap(enw[b]);
            T0.l[u0] = -1; T0.r[u0] = -1; T0.lab[u0] = nl;
            T1.l[u1] = -1; T1.r[u1] = -1; T1.lab[u1] = nl;
            any = true;
        }
    }
    if (!any) return false;

    // Collect kernel leaf-labels by traversing from the root: only REACHABLE
    // leaves count. (Collapsed cherries leave orphaned child nodes behind that
    // still look like leaves; scanning all nodes would double-count them.)
    vector<int> present;
    {
        vector<unsigned char> seen(next_label + 1, 0);
        vector<int> stk; stk.push_back(T0.root);
        while (!stk.empty()) {
            int u = stk.back(); stk.pop_back();
            if (T0.l[u] == -1) { int v = T0.lab[u]; if (!seen[v]) { seen[v] = 1; present.push_back(v); } }
            else { stk.push_back(T0.l[u]); stk.push_back(T0.r[u]); }
        }
    }
    sort(present.begin(), present.end());
    int K = (int)present.size();
    vector<int> relabel(next_label + 1, -1);
    expand_nwk.assign(K, string());
    for (int i = 0; i < K; ++i) { relabel[present[i]] = i; expand_nwk[i] = enw[present[i]]; }

    out0 = T0.serialize(T0.root, relabel) + ";";
    out1 = T1.serialize(T1.root, relabel) + ";";
    return true;
}

// Expand a kernel-labelled component newick to original labels by substituting
// each maximal digit-run token t (1-based kernel label) with expand_nwk[t-1].
inline string expand_component(const string& knw, const vector<string>& expand_nwk) {
    string out;
    out.reserve(knw.size() * 2 + 8);
    size_t i = 0, m = knw.size();
    while (i < m) {
        if (isdigit((unsigned char)knw[i])) {
            size_t j = i;
            while (j < m && isdigit((unsigned char)knw[j])) ++j;
            int t = std::stoi(knw.substr(i, j - i));
            out += expand_nwk[t - 1];
            i = j;
        } else { out.push_back(knw[i]); ++i; }
    }
    return out;
}

} // namespace kern

// Process-wide exit status produced by run_solver(); main() returns it after
// the worker thread joins.
static std::atomic<int> g_solver_exit_code{0};

// The entire solving pipeline (input parse -> kernelize -> Solver -> flush).
// Run on a large-stack worker thread (see main) so that a deeply nested
// (caterpillar / ladder) input cannot overflow the default ~8 MB stack in any
// recursive routine -- the kernel Newick parser/serializer, restricted_newick,
// etc. Such an overflow is an UNCATCHABLE SIGSEGV that emits nothing, which
// under PACE's "one infeasible output disqualifies the whole submission" rule
// is the worst possible outcome. A 1 GiB stack lets these recursions run to
// millions of levels deep; only touched pages become resident, so this costs
// no real memory on the shallow trees that are the common case.
static void run_solver() {
    int t = -1, n = -1;
    vector<string> trees;
    string line;
    try {
        while (getline(cin, line)) {
            line = trim(line);
            if (line.empty()) continue;
            if (line.rfind("#p", 0) == 0) {
                string tag;
                stringstream ss(line);
                ss >> tag >> t >> n;
                // Install a valid singleton fallback the moment n is known, so a
                // SIGTERM during the remaining input read still flushes a valid
                // (worst-case k=n) forest instead of nothing. (Closes a small
                // pre-existing empty-output window; also present in submission.)
                if (n > 0) {
                    vector<string> triv;
                    triv.reserve(n);
                    for (int i = 1; i <= n; ++i) triv.push_back(to_string(i) + ";");
                    best_cache::install(triv);
                }
                continue;
            }
            if (line[0] == '#') continue;
            trees.push_back(line);
        }
        if (t < 0 || n <= 0 || (int)trees.size() < t) throw runtime_error("missing #p or tree lines");
        if (t != 2) {
            // Heuristic track is two-tree MAF. For non-two-tree input, return the always-feasible singleton partition.
            for (int i = 1; i <= n; ++i) cout << i << ";\n";
            g_solver_exit_code.store(0, std::memory_order_relaxed);
            return;
        }

        // Publish the trivial singleton forest BEFORE constructing the Solver,
        // so that even if construction throws or hangs the signal handler has
        // something valid to emit.
        {
            vector<string> trivial;
            trivial.reserve(n);
            for (int i = 1; i <= n; ++i) trivial.push_back(to_string(i) + ";");
            best_cache::install(trivial);
        }

        // ---- Lossless common-subtree kernelization (DEV) --------------------
        // Collapse common pendant subtrees, solve the smaller kernel, expand at
        // output. Optimality-preserving, so this cannot raise the true optimum;
        // any hiccup falls back to solving the full instance. PACE_NOKERNEL=1
        // disables it (for A/B against the baseline).
        string knw0 = trees[0], knw1 = trees[1];
        int solve_n = n;
        vector<string> expand_nwk;
        if (std::getenv("PACE_NOKERNEL") == nullptr) {
            string r0, r1; vector<string> em;
            bool ok = false;
            try { ok = kern::kernelize(trees[0], trees[1], n, r0, r1, em); }
            catch (...) { ok = false; }
            // Adopt the kernel only if it achieved a MEANINGFUL reduction (>=10%).
            // A barely-reduced kernel (e.g. a time-capped partial reduction of a
            // deep common ladder) is counterproductive: it costs build time yet
            // stays nearly as large, delaying the full solver's O(n) common-clade
            // sweep that collapses the shared structure directly. On such inputs
            // solving the full instance is both faster and better.
            if (ok && !em.empty() && (int)em.size() <= (int)(0.90 * (double)n)) {
                knw0 = std::move(r0); knw1 = std::move(r1);
                solve_n = (int)em.size(); expand_nwk = std::move(em);
            }
        }
        if (std::getenv("PACE_KDEBUG")) std::cerr << "[kernel] n=" << n << " K=" << solve_n << "\n";
        Solver solver(solve_n, knw0, knw1);
        if (!expand_nwk.empty()) { solver.expand_nwk_ = std::move(expand_nwk); solver.orig_n_ = n; }
        solver.solve();
        if (std::getenv("PACE_KDEBUG"))
            std::cerr << "[exact] calls=" << exactmaf::dbg_calls()
                      << " done=" << exactmaf::dbg_done()
                      << " capped=" << exactmaf::dbg_capped()
                      << " maxm_done=" << exactmaf::dbg_maxm_done() << "\n";

        // Normal exit: prevent a late SIGTERM from interleaving handler output
        // with buffered iostream output. Print through the same single cached
        // write(2)-based path used by the signal handler.
        {
            struct sigaction sa;
            std::memset(&sa, 0, sizeof(sa));
            sa.sa_handler = SIG_IGN;
            sigemptyset(&sa.sa_mask);
            ::sigaction(SIGTERM, &sa, nullptr);
            ::sigaction(SIGINT,  &sa, nullptr);
        }
        best_cache::install(solver.best_output_lines());
        best_cache::flush_to_stdout_safely();
    } catch (const exception& e) {
        // Never print diagnostics to stdout: stdout is the solution channel.
        cerr << "solver error: " << e.what() << "\n";
        if (n > 0) {
            for (int i = 1; i <= n; ++i) cout << i << ";\n";
            g_solver_exit_code.store(0, std::memory_order_relaxed);
            return;
        }
        g_solver_exit_code.store(1, std::memory_order_relaxed);
        return;
    }
    g_solver_exit_code.store(0, std::memory_order_relaxed);
}

extern "C" void* solver_thread_entry(void*) {
    run_solver();
    return nullptr;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // ---- 5-minute budget with safety margin ---------------------------------
    // PACE sends SIGTERM at the 5-minute mark. We target 4:58 so cooperative
    // checks use almost all available time; if the kernel still delivers
    // SIGTERM, the handler will flush the cached best-so-far via write(2).
    deadline::init(configured_time_budget_seconds());

    // Install signal handlers BEFORE doing any work so an early SIGTERM is
    // caught cleanly. SIGINT is also handled so Ctrl-C during local testing
    // produces the same behavior. Handlers are process-global, so they fire
    // regardless of which thread the SIGTERM is delivered to; the handler only
    // reads atomics and calls write(2)/_exit, so running on any thread is safe.
    {
        struct sigaction sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sa_handler = pace_signal_handler;
        // SA_RESTART would resume read()/write() across the signal, but here
        // we want the handler to immediately _exit, so it doesn't matter.
        sigemptyset(&sa.sa_mask);
        ::sigaction(SIGTERM, &sa, nullptr);
        ::sigaction(SIGINT,  &sa, nullptr);
    }

    // Run the whole solver on a worker thread with a large (1 GiB) stack so deep
    // recursive routines cannot overflow the default stack and SIGSEGV with
    // empty output (see run_solver). pthread_attr_setstacksize sizes the stack
    // independently of `ulimit -s`, so this is robust even under an 8 MB soft
    // limit. If thread creation fails for any reason, fall back to running on
    // the main stack rather than not running at all.
    bool ran_on_thread = false;
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) == 0) {
        const size_t big_stack = (size_t)1 << 30;  // 1 GiB
        if (pthread_attr_setstacksize(&attr, big_stack) == 0) {
            pthread_t tid;
            if (pthread_create(&tid, &attr, solver_thread_entry, nullptr) == 0) {
                pthread_join(tid, nullptr);
                ran_on_thread = true;
            }
        }
        pthread_attr_destroy(&attr);
    }
    if (!ran_on_thread) run_solver();
    return g_solver_exit_code.load(std::memory_order_relaxed);
}
