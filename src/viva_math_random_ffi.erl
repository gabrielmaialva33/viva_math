%%%-------------------------------------------------------------------
%%% @doc viva_math/random FFI wrapper.
%%%
%%% Thin shim around the OTP `rand` module so that the Gleam side can
%%% expose a pure, immutable `Seed` value. We pin the algorithm to
%%% `exsss` (the OTP 22+ default Xorshift116** with StarStar scrambler)
%%% which is fast on 64-bit BEAM and has good statistical properties.
%%% @end
%%%-------------------------------------------------------------------
-module(viva_math_random_ffi).

-export([
    seed_default/1,
    seed_with_algo/2,
    uniform_real/1,
    uniform_int/2,
    normal_standard/1,
    normal_with/3,
    jump/1,
    export_seed/1
]).

%% @doc Build an initial seed from an integer using the default algorithm.
seed_default(N) when is_integer(N) ->
    rand:seed_s(exsss, N).

%% @doc Build a seed with an explicit algorithm atom.
%% Valid algorithms: exsss | exro928ss | exrop | exsp | exs1024s | mwc59 | default.
seed_with_algo(Algo, N) when is_atom(Algo), is_integer(N) ->
    rand:seed_s(Algo, N).

%% @doc Generate a uniform real in [0.0, 1.0).
uniform_real(S) ->
    rand:uniform_real_s(S).

%% @doc Generate a uniform integer in [1, N].
uniform_int(N, S) when is_integer(N), N >= 1 ->
    rand:uniform_s(N, S).

%% @doc Standard normal sample N(0, 1).
normal_standard(S) ->
    rand:normal_s(S).

%% @doc Normal sample N(Mu, Sigma^2).
%% NOTE: rand:normal_s/3 expects variance (sigma^2), but our API accepts
%% standard deviation. We square here so callers can pass sigma naturally.
normal_with(Mu, Sigma, S) when is_number(Sigma), Sigma >= 0 ->
    rand:normal_s(Mu, Sigma * Sigma, S).

%% @doc Advance the state by 2^64 calls (non-overlapping streams).
jump(S) ->
    rand:jump(S).

%% @doc Export the seed to a portable representation.
export_seed(S) ->
    rand:export_seed_s(S).
