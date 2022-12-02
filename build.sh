echo "Configuring and building thirdparty/fast ..."

cd thirdparty/fast
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../Sophus

echo "Configuring and building thirdparty/Sophus ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../g2o

echo "Configuring and building thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../../

echo "Configuring and building HSO ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
