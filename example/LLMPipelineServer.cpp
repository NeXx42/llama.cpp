#include <iostream>
#include <SessionManager.h>

nexx::SessionManager manager;

int main(int argc, char ** argv) {
    manager = nexx::SessionManager();
    return manager.Setup();
}