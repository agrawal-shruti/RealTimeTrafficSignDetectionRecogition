chmod 777 sign_mser.sh
g++ norm2.cpp -o norm2 `pkg-config --cflags --libs opencv`
./norm2
