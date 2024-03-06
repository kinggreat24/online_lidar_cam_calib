#include <cmath>
#include <random>

#include <matplot/matplot.h>

int main() {
    using namespace matplot;

    std::vector<double> x = randn(10000, 0, 1);

    auto h = hist(x);
    std::cout << "Histogram with " << h->num_bins() << " bins" << std::endl;

    show();
    return 0;
}