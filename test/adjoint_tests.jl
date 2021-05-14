push!(LOAD_PATH, "./src/simutils/")

using Simutils
using Test
using Random
using ChainRulesTestUtils
using ChainRulesCore
using Infiltrator
using DiffEqOperators

Random.seed!(1)

ChainRulesCore.debug_mode() = true


@testset "wave simulation tests" begin
    
    # @testset "excitation function" begin
    #     vector_length = 100
    #     number_of_excitation_functions = 10
    #     a = randn(vector_length)
    #     positions = rand(1:vector_length, number_of_excitation_functions)

    #     function_array = [t -> rand() for i in 1:number_of_excitation_functions]

    #     test_rrule(vector_excitation, a, positions ⊢ DoesNotExist(), function_array ⊢ DoesNotExist() , 1.0 ⊢ DoesNotExist());
    # end

    @testset "RightStaggeredDifference" begin
        domain_length = 1.
        dx = 0.01
        number_of_cells = Int(div(domain_length, dx))
        coeffs = rand(number_of_cells)
        # Ax = RightStaggeredDifference{1}(1, 2, dx, number_of_cells, coeffs)
        test_rrule(RightStaggeredDifference{1}, 1 ⊢ DoesNotExist(), 2 ⊢ DoesNotExist() , 
                    dx ⊢ DoesNotExist(), number_of_cells ⊢ DoesNotExist(), coeffs ⊢ DoesNotExist(); check_inferred=false)
    end
end

