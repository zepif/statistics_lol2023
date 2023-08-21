#include <bits/stdc++.h>
using namespace std;
#define ll long long
int main() {
    int num;
    char a;
    while (cin >> a) {
        cin >> num;
        cout << "\"" << num << "\"";
        cin >> a;
        cin >> a;
        cin >> num;
        cout << " : " << "\"" << num << "\"" << ",\n";
        cin >> a;
    }
    return 0;
}