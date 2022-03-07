#include "NN.h"

//... data generator code here

typedef std::vector<RowVector *> data;
int main()
{
    NN n({2, 3, 1});
    data in, out;
    genData("test");
    ReadCSV("test-in", in);
    ReadCSV("test-out", out);
    n.train(in, out);
    return 0;
}