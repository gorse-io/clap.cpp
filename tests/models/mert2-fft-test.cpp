// Exercise the production FFT against a double direct DFT, not another FFT.
#include "clap-internal.h"
#include <algorithm>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
    constexpr std::size_t n=2048;
    const double pi=std::acos(-1.0);
    std::vector<std::complex<float>> input(n);
    for(std::size_t i=0;i<n;++i) {
        const double t=(176.0+i)/24000.;
        const float pcm=static_cast<float>(.19*std::sin(2*pi*(220*t+31*t*t))+.03*std::cos(2*pi*443*t));
        const float window=static_cast<float>(.5-.5*std::cos(2*pi*i/n));
        input[i]=pcm*window;
    }
    std::vector<std::complex<double>> transformed(input.begin(), input.end());
    clap::fft(transformed);
    std::vector<std::complex<float>> actual(transformed.begin(), transformed.end());
    double complex_error=0,log_error=0;
    for(std::size_t k=0;k<=n/2;++k) {
        std::complex<double> reference{};
        for(std::size_t i=0;i<n;++i) reference+=static_cast<double>(input[i].real())*std::polar(1.,-2*pi*double(k*i)/n);
        const std::complex<float> expected(reference);
        complex_error=std::max(complex_error,double(std::abs(actual[k]-expected)));
        const double a=10*std::log10(std::max(double(std::norm(actual[k])),1e-10));
        const double b=10*std::log10(std::max(double(std::norm(expected)),1e-10));
        log_error=std::max(log_error,std::abs(a-b));
    }
    std::cout<<"FFT complex max_abs="<<complex_error<<" log_power max_abs_db="<<log_error<<'\n';
    if(complex_error>2e-5 || log_error>1e-3) return 1;
    std::vector<std::complex<double>> singleton{{2, -3}};
    clap::fft(singleton);
    if (singleton[0] != std::complex<double>(2, -3)) return 1;
    for (std::size_t size : {std::size_t{0}, std::size_t{3}}) {
        std::vector<std::complex<double>> invalid(size);
        try { clap::fft(invalid); return 1; }
        catch (const std::invalid_argument &) {}
    }
}
