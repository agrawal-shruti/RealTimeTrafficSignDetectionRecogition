chmod 777 norm.sh
g++ norm.cpp -o norm `pkg-config --cflags --libs opencv`
./norm
