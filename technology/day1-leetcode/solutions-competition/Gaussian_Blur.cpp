#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int main(){
    int M,N,R,C;
    cin >> M >> N >> R >> C;
    vector<vector<int>> v;
    for(int i=0; i < M;i++){
        vector<int> temp;
        for(int j=0; j < N;j++){
            int x;
            cin >> x;
            temp.push_back(x);
        }
        v.push_back(temp);
    }

    int T;
    cin >> T;
    vector<pair<int,int>> parovi;
    for(int i=0;i<T;i++){
        int x1,x2;
        cin >> x1 >> x2;
        parovi.push_back(make_pair(x1,x2));
    }

    
    for(int t=0;t<T;t++){
        long long sum=0;
        for(int i=-R;i<=R;i++){
            for(int j = -R;j<=R;j++){
                long long clan = v[parovi[t].first+i][parovi[t].second+j]*pow(C,abs(i)+abs(j));
                sum = sum + clan%(1000000007);
            }
        }
        cout << sum << endl;
    }
    
}