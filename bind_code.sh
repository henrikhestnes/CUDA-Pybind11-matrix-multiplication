git submodule init
git submodule update
mkdir -p build
cd build
cmake --build .. --target clean
cmake .. -DCMAKE_C_COMPILER=$(which gcc-8) -DCMAKE_CXX_COMPILER=$(which g++-8)
make all
cd ..
