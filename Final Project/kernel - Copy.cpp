#include <mat.h>
#include <matrix.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>



 void matread(const char* file, double* pr)
{
    // open MAT-file
    MATFile* pmat = matOpen(file, "r");
    std::vector<double> v;
    if (pmat == NULL) return;

    // extract the specified variable
    mxArray* arr = matGetVariable(pmat, "xAxis");
    if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {
        // copy data
        mwSize num = mxGetNumberOfElements(arr);
        pr = mxGetPr(arr);
        
        for (int i = 0;i < num;i++) {
            std::cout << &pr[i] << " ";
        }
        //if (pr != NULL) {
        //    v.reserve(num); //is faster than resize :-)
        //    v.assign(pr, pr + num);
        //}
    }

    // cleanup
    mxDestroyArray(arr);
    matClose(pmat);
}




 int main()
{
    const char* file = "G:\Ji Chen's Lab\Chen's Lab\ActiveMRIHeating\Bioheat Equation\Material_Map.mat";
    double* v = NULL;
    matread(file, v);


    return 0;
}


