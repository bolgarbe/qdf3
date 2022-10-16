#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>
#include <string>

class Exception:public std::exception
{
public:
    Exception(const char* msg):msg_(msg){}

    const char* what() const throw()
    {
        return msg_.c_str();
    }

private:
    std::string msg_;
};

#endif // EXCEPTIONS_H
