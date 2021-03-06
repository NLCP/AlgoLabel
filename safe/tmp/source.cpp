using ll = long long;
const int INF = 0x3f3f3f3f;
const ll LINF = 0x3f3f3f3f3f3f3f3fLL;
const double EPS = 1e-8;
const int MOD = 1000000007;
const int dy[] = {1, 0, -1, 0}, dx[] = {0, -1, 0, 1};
const int dy8[] = {1, 1, 0, -1, -1, -1, 0, 1}, dx8[] = {0, -1, -1, -1, 0, 1, 1, 1};
template <typename T, typename U> inline bool chmax(T &a, U b) { return a < b ? (a = b, true) : false; }
template <typename T, typename U> inline bool chmin(T &a, U b) { return a > b ? (a = b, true) : false; }
struct IOSetup {
IOSetup() {
cin.tie(nullptr);
ios_base::sync_with_stdio(false);
cout << fixed << setprecision(20);
}
} iosetup;
void solve() {
int n; cin >> n;
vector<string> s(n); for(int i=(0);i<(n);++i) cin >> s[i];
for(int i=(0);i<(n);++i) for(int j=(0);j<(n);++j) {
if (s[i][j] == '1') {
if (j + 1 == n || s[i][j + 1] == '1') continue;
if (i + 1 == n || s[i + 1][j] == '1') continue;
cout << "NO\n";
return;
}
}
cout << "YES\n";
}
int main() {
int t; cin >> t;
while (t--) solve();
return 0;
}