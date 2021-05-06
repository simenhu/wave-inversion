push!(LOAD_PATH, "./src/simutils/")
using Simutils
using Test
using Random
using ChainRulesTestUtils
# using ChainRulesCore
using Infiltrator

Random.seed!(1)

ChainRulesCore.debug_mode() = true


@testset "wave simulation tests" begin
    @testset "excitation function" begin
        vector_length = 100
        number_of_excitation_functions = 10
        a = randn(vector_length)
        positions = rand(1:vector_length, number_of_excitation_functions)

        function_array = [t -> rand() for i in 1:number_of_excitation_functions]

        test_rrule(vector_excitation, a, positions ⊢ DoesNotExist(), function_array ⊢ DoesNotExist() , 1.0 ⊢ DoesNotExist());
    end
end

