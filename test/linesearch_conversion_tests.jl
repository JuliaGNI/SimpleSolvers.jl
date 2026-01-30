using SimpleSolvers

function allocate_linesearch_methods(T::DataType)
    st = Static(T; α=one(T))
    bt = Backtracking(T)
    qu = Quadratic(T; ε=T(1e-5)) # here this constant is specified manually as it otherwise depends on the DataType used
    bq = BierlaireQuadratic(T)
    bi = Bisection(T)
    st, bt, qu, bq, bi
end

function convert_linesearches_test(T₁::DataType, T₂::DataType; rtol=T₂(1e-3))
    st₁, bt₁, qu₁, bq₁, bi₁ = allocate_linesearch_methods(T₁)
    st₂, bt₂, qu₂, bq₂, bi₂ = allocate_linesearch_methods(T₂)

    @test ≈(st₂, convert(T₂, st₁); rtol=rtol)
    @test ≈(bt₂, convert(T₂, bt₁); rtol=rtol)
    @test ≈(qu₂, convert(T₂, qu₁); rtol=rtol)
    @test ≈(bq₂, convert(T₂, bq₁); rtol=rtol)
    @test ≈(bi₂, convert(T₂, bi₁); rtol=rtol)

    nothing
end

convert_linesearches_test(Float32, Float64)
convert_linesearches_test(Float64, Float32)
