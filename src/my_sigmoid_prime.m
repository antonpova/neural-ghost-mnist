function res = my_sigmoid_prime(x)
    %Sigmoid: 1/(1+e^(-x))
    ex = exp(x);
    res = ex./((ex+1).^2);
end