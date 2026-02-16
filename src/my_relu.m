function res = my_relu(x)
    mask = (x>0);
    res = x.*mask; 
end