%%%-------------------------------------------------------------------
%%% @doc Test-only IEEE-754 helpers.
%%% @end
%%%-------------------------------------------------------------------
-module(test_support_ffi).

-export([ulp_distance/2]).

ulp_distance(A, B) when is_float(A), is_float(B) ->
    IA = ordered_float_bits(A),
    IB = ordered_float_bits(B),
    abs(IA - IB).

ordered_float_bits(X) ->
    <<Bits:64/unsigned-integer>> = <<X:64/float>>,
    case Bits band 16#8000000000000000 of
        0 ->
            Bits;
        _ ->
            -1 - (Bits bxor 16#8000000000000000)
    end.
