    
if !(@isdefined SSCONFIG)
    const global SSCONFIG = Dict()
end

function add_config(name, value)
    if haskey(SSCONFIG, name)
        @warn("Overwriting parameter $name.")
    end
    SSCONFIG[name] = value
end

function set_config(name, value)
    if !haskey(SSCONFIG, name)
        @warn("Unknown parameter name: adding parameter.")
    end
    SSCONFIG[name] = value
end

function get_config(name)
    if haskey(SSCONFIG, name)
        return SSCONFIG[name]
    else
        @warn("Unknown parameter name.")
        return nothing
    end
end

function get_config_dictionary()
    return Dict(SSCONFIG)
end
