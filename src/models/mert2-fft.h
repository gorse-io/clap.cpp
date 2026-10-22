#pragma once
#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>
namespace clap::detail {
// F32 inputs/outputs, double intermediates: repeated float twiddle products
// raise the low-energy spectral floor, amplified by log-mel normalization.
inline void mert2_fft(std::vector<std::complex<float>> &output) {
    constexpr double pi=3.141592653589793238462643383279502884;
    std::vector<std::complex<double>> values(output.begin(),output.end());
    const std::size_t size=values.size();
    for(std::size_t i=1,j=0;i<size;++i) {
        std::size_t bit=size>>1;
        for(;j&bit;bit>>=1) j^=bit;
        j^=bit;
        if(i<j)std::swap(values[i],values[j]);
    }
    for(std::size_t width=2;width<=size;width<<=1) {
        const auto root=std::polar(1.,-2*pi/static_cast<double>(width));
        for(std::size_t offset=0;offset<size;offset+=width) {
            std::complex<double> factor(1.);
            for(std::size_t index=0;index<width/2;++index) {
                const auto left=values[offset+index];
                const auto right=values[offset+index+width/2]*factor;
                values[offset+index]=left+right;
                values[offset+index+width/2]=left-right;
                factor*=root;
            }
        }
    }
    for(std::size_t i=0;i<size;++i)output[i]=static_cast<std::complex<float>>(values[i]);
}
} // namespace clap::detail
