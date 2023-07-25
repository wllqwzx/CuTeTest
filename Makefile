.PHONY: build

build:
	rm -f ./main
	nvcc -O3 -std=c++17 main.cu -o main -I./cutlass/include \
		 -gencode arch=compute_80,code=sm_80 \
		 -gencode arch=compute_86,code=sm_86 \
		 -gencode arch=compute_87,code=sm_87 \
		 --ptxas-options=-v --expt-relaxed-constexpr
	./main
