close all;

numer_albumu=321358; %% Wpisz swój numer albumu 
rng(numer_albumu);
N=32;
items(:,1)=round(0.1+0.9*rand(N,1),1);
items(:,2)=round(1+99*rand(N,1));

max_weight = 0.3*sum(items(:,2));

test_algorythm(items, max_weight, 'one point', 0.01);


% function used to test the algotythm
function test_algorythm(items, max_weight, crossover_type, mutation_chance)
    [best_solution, fitness_history, offspring_count]  = genetic_knapsack(items, max_weight , mutation_chance, crossover_type);
    disp(['Mutation chance: ', num2str(mutation_chance)])
    disp(['Crossover type: ', crossover_type])
    disp(['Best solution:', num2str(best_solution)]);
    disp(['Best solution value: ', num2str(best_solution*items(:,1))]);
    disp(['Best solutin weight: ', num2str(best_solution*items(:,2))]);
    disp(['mean crossover offspring: ', num2str(mean(offspring_count(:,1)))])
    disp(['mean elite offspring: ', num2str(mean(offspring_count(:,2)))])
    disp(['mean mutated offspring: ', num2str(mean(offspring_count(:,3)))])
    
    plot_fitness(fitness_history)
end

% genetic algorythm
function [best_solution, fitness_history, offspring_count] = genetic_knapsack(items, max_weight, mutation_chance, crossover_type)
    % parameters
    population_size = 50;
    num_generations = 10000;
    crossover_rate = 0.95;
    patience = 50; 
    improvement_threshold = 1e-3;
  
    % initialization
    population = randi([0, 1], population_size, size(items, 1));
    fitness_history = zeros(num_generations, 5);
    offspring_count = zeros(num_generations, 3);
    best_fitness = -Inf;
    generations_without_improvement = 0;  

    for generation = 1:num_generations
        fitness = evaluate(population, items, max_weight);

        [current_best_fitness, ~] = max(fitness);
        fitness_history(generation, 1) = current_best_fitness;
        fitness_history(generation, 2) = min(fitness);
        fitness_history(generation, 3) = mean(fitness);
        fitness_history(generation, 4) = var(fitness);

        % stopping
        if current_best_fitness > best_fitness + improvement_threshold
            best_fitness = current_best_fitness;
            generations_without_improvement = 0; 
        else
            generations_without_improvement = generations_without_improvement + 1;
        end

        if generations_without_improvement >= patience
            fitness_history = fitness_history(1:(generation-patience+10), 1:4);
            offspring_count = offspring_count(1:(generation-patience+10), 1:3);
            break;
        end

        offspring_count(generation, 1) = 0;
        offspring_count(generation, 2) = 0;
        offspring_count(generation, 3) = 0;
        
        % Selection
        selected_indices = tournament_selection(fitness, population_size);
        selected_population = population(selected_indices, :);

        % crossover
        if crossover_type == 'two point'
            % two point crossover
            new_population = zeros(size(population));
            for i = 1:2:population_size
                parent1 = selected_population(randi([1, population_size]), :);
                parent2 = selected_population(randi([1, population_size]), :);
                if rand < crossover_rate 
                    points = sort(randperm(size(items, 1), 2));
                    new_population(i, :) = [parent1(1:points(1)), parent2(points(1)+1:points(2)), parent1(points(2)+1:end)];
                    new_population(i + 1, :) = [parent2(1:points(1)), parent1(points(1)+1:points(2)), parent2(points(2)+1:end)];
                    offspring_count(generation, 1) = offspring_count(generation, 1) + 2;
                else % 20% szansa na brak krzyżowania (kopiowanie rodziców)
                    new_population(i, :) = parent1;
                    new_population(i + 1, :) = parent2;
                    offspring_count(generation, 2) = offspring_count(generation, 2) + 2;
                end
            end
        else 
            % one point crossover
            new_population = zeros(size(population));
            for i = 1:2:population_size
                parent1 = selected_population(randi([1, population_size]), :);
                parent2 = selected_population(randi([1, population_size]), :);
                if rand < crossover_rate
                    point = randi([1, size(items, 1)-1]);
                    new_population(i, :) = [parent1(1:point) parent2(point+1:end)];
                    new_population(i+1, :) = [parent2(1:point) parent1(point+1:end)];
                    offspring_count(generation, 1) = offspring_count(generation, 1) + 2;
                else
                    new_population(i, :) = parent1;
                    new_population(i+1, :) = parent2;
                    offspring_count(generation, 2) = offspring_count(generation, 2) + 2;
                end
            end
        end

        % Mutation
        for i = 1:population_size
            mutated = false;
            for j = 1:size(items, 1)
                if rand < mutation_chance
                    new_population(i, j) = ~new_population(i, j);
                    mutated = true;
                end
            end
            if mutated
                offspring_count(generation, 3) = offspring_count(generation, 3) + 1;
            end
        end
        population = new_population;
    end
    
    fitness = evaluate(population, items, max_weight);
    [~, best_index] = max(fitness);
    best_solution = population(best_index, :);
end

% function generating plots
function plot_fitness(fitness_history)
    figure;
    subplot(2,2,1)
    plot(fitness_history(:,1), 'Linewidth', 1)
    xlabel('Generation');
    ylabel('Fitness');
    title('Fitness maximum value')
    grid on;
    subplot(2,2,2)
    plot(fitness_history(:,2), 'Linewidth', 1)
    xlabel('Generation');
    ylabel('Fitness');
    title('Fitness minimum value')
    grid on;
    subplot(2,2,3)
    plot(fitness_history(:,3), 'Linewidth', 1)
    xlabel('Generation');
    ylabel('Mean fitness');
    title('Fitness mean value')
    grid on;
    subplot(2,2,4)
    plot(fitness_history(:,4), 'Linewidth', 1)
    xlabel('Generation');
    ylabel('Fitness variance');
    title('Fitness variance value')
    grid on;
end

% fitness function
function fitness = evaluate(population, items, max_weight)
    values = population * items(:, 1);
    weights = population * items(:, 2);
    penalty = max(0, weights - max_weight); 
    fitness = values - penalty;

end

% tournament selection function
function selected_indices = tournament_selection(fitness, population_size)
    tournament_size = 3;
    selected_indices = zeros(population_size, 1);
    for i = 1:population_size
        competitors = randi([1, population_size], tournament_size, 1);
        [~, best] = max(fitness(competitors));
        selected_indices(i) = competitors(best);
    end
end

