set -xe

rm -rf build microrts.jar
mkdir build
javac -encoding ISO-8859-1 -d "./build" -cp "./lib/*" -sourcepath "./src"  $(find ./src/* | grep .java)
# javac ./src/tests/*.java ./src/tests/sockets/*.java
cp -a lib/. build/
cd build
find . -maxdepth 1 -type f -name "*jar" -exec jar -xf {} \;
jar cvf microrts.jar *
mv microrts.jar ../microrts.jar
cd ..
rm -rf build