#include <iostream>
using namespace std;



int main() {


    int length = 192;
    int width = 256;
    int size_kernel = 31;
    int width_fil = (width - size_kernel + 1);
    int size_fil = length * width_fil;
    int index0 = 0;
    
    //int a = 114;
    //double da = double(a);
    //double dsize = double(width_fil);
    ////double r = remainder(da, dsize);
    //int r = a % width_fil;
    //int quo = a / width_fil * width;
    //int i0 = quo + r;

    for (int index1 = 0; index1 < 1000; index1++) {

        for (int j = 0; j < size_kernel; j++) {

            index0 = index1 % width_fil + index1 / width_fil * width + j;
            

        }
        cout << "index1 now is: " << index1 << " ,index0 now is: " << index0 << endl;
    }

    return 0;
}

