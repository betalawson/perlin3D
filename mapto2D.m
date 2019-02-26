function points2D = mapto2D(vertices, edgematrix)
% This function takes an input set of vertices in 3D space, as well as
% a list of the edges connecting them, and maps to an equivalent 2D space
% for the purposes of texture mapping

% Read out the number of points
Np = size(edgematrix,1);

% Initialise the procedure by just taking the x and y co-ordinates of each
% of the points
points2D = [vertices(:,1), vertices(:,2)];

% Plot the current state
scatter(points2D(:,1), points2D(:,2), 15, 'filled');
drawnow;

% Define optimisation parameters
tol = 1e-6;
max_iters = 100;
step_accept_ratio = 0.9;
alpha_min = 1e-6;

% The mapping to two-dimensional space is found by optimising the energy of
% a set of springs between the vertices in 3D. The conjugate gradient
% method is used to perform this optimisation, because the derivative is
% available

% First, calculate some starting information for the method
E = calcEnergy(vertices, points2D, edgematrix);
E_old = E
dE = calcDeriv(vertices, points2D, edgematrix);
dE_old = dE;

% For conjugate gradient, the initial search direction is just steepest
% descent direction
d = -dE;

stop = 0;
iters = 0;
while ~stop
    
    % Find an acceptable step length in this direction
    stepped = 0;
    alpha = 10;
    while ~stepped
        
        % Calculate the energy associated with this step
        E_step = calcEnergy(vertices, points2D + alpha * reshape(d,2,Np)', edgematrix);
        
        if ( E_step / E < step_accept_ratio ) || ( alpha <= alpha_min )
            stepped = 1;
        else
            alpha = alpha * 1/2;
        end
        
    end
        
    % Update the system using the found step length
    points2D = points2D + alpha * reshape(d,2,Np)';
    E = E_step;
    dE = calcDeriv(vertices, points2D, edgematrix);
    
    % Calculate the new search direction
    beta = max([0, dE * (dE - dE_old)' / ( dE_old * dE_old')]);
    d = -dE + beta * d;
        
    % Increment iteration count
    iters = iters + 1;
    
    % Terminate loop if maximum iterations reached, or improvement too slow
    if iters > max_iters || abs(1 - E/E_old) < tol
        stop = 1;
    end
    
    % Update 'old' value of energy to the new value in preparation for new
    % loop, and ditto for the derivative
    E_old = E;
    dE_old = dE;
    
    % Plot the current state
    scatter(points2D(:,1), points2D(:,2), 15, 'filled');
    drawnow;
    
end

end

function E = calcEnergy(vertices, points2D, edgematrix)
% This subfunction calculates the energy to be minimised

% Read out the number of points
Np = size(edgematrix,1);

% Initialise energy at zero
E = 0;

% Calculate total energy as a loop over all connected nodes (or
% equivalently, a set of springs between all connected nodes)
for j = 1:Np
    for i = find(edgematrix(j,:))

        % From formula in Mailot 1993
        E = E + 2 * ( norm( points2D(j,:) - points2D(i,:) )^2 - norm( vertices(j,:) - vertices(i,:) )^2 )^2 / ( norm( vertices(j,:) - vertices(i,:) )^2 );
        
    end
end


end

function dE = calcDeriv(vertices, points2D, edgematrix)
% This subfunction calculates the Jacobian of the minimisation problem. As
% there is only one function to minimise, this is a row matrix

% Read out the number of points
Np = size(edgematrix,1);

% Initialise as zero matrix
dE = zeros(1,Np*2);

% Loop over each connection between nodes
for j = 1:Np
    for i = find(edgematrix(j,:))
        
        % From formula in Mailot 1993
        dE(2*j-1) = dE(2*j-1) + 8 * (norm(points2D(j,:) - points2D(i,:))^2 - norm(vertices(j,:) - vertices(i,:))^2 ) / ( norm(vertices(j,:) - vertices(i,:))^2 ) * (points2D(j,1) - points2D(i,1));
        dE(2*j) = dE(2*j) + 8 * (norm(points2D(j,:) - points2D(i,:))^2 - norm(vertices(j,:) - vertices(i,:))^2 ) / ( norm(vertices(j,:) - vertices(i,:))^2 ) * (points2D(j,2) - points2D(i,2));
        
    end 
end


end