using LinearAlgebra, Printf, SparseArrays

function chol_tri(d, e)
    n = length(d)
    # G = zeros(n, n)
    d[1] = sqrt(d[1])
    for j in 1:n
        for i in j:n
            if (abs(i - j) <= 1)
                if i != j
                    e[j] = e[j] / d[j]
                elseif i == j && i != 1
                    d[i] = sqrt(d[i] - e[i - 1]^2)
                end
            end
        end
    end
    return d, e
end

function resolve_chol_tri(d, e, b)
    n = length(d)
    chol_tri(d, e)
    
    # b = G y
    ### y = G\b
    b[1] = b[1] / d[1]
    for i in 2:n
        b[i] = (b[i] - e[i - 1] * b[i - 1]) / d[i]
    end
    
    # y = G^T x
    ### x = G'\y
    b[4] = b[4] / d[4]
    for j in (n-1):-1:1
        b[j] = (b[j] - e[j] * b[j + 1]) / d[j]
    end
    
    return b
end

function main()
  n = 100
  E_tri = 0.0
  E_sis = 0.0
  all = 0
  contador = 0
  for n = [5; 10; 50; 100; 500; 1000; 5000; 10000; 50000; 100000]
    @printf("Avaliando com n = %7d", n)
    Δt = time()
    failed = false
    for t = 1:100
      t % 5 == 0 && print(".")
      contador += 1
      d = 10 .+ rand(n)
      e = randn(n-1)

      A = spdiagm(0 => d, -1 => e, 1 => e)
      b = A * ones(n)

      all += @allocated chol_tri(d, e)
      G = spdiagm(0 => d, 1 => e)
      all += @allocated resolve_chol_tri(d, e, b)

      E_tri += norm(G' * G - A)
      E_sis += norm(b .- 1)
      if time() - Δt > 10
        failed = true
        break
      end
    end
    if !failed
      @printf(" Pronto. Erros até agora: %8.2e  %8.2e\n", E_tri / contador, E_sis / contador)
    end
    Δt = time() - Δt
    if Δt > 10
      println("\033[31mExcesso de tempo ocorrido, verifique que seu código está otimizado\033[0m")
      break
    end
  end
  E_tri /= contador
  E_sis /= contador
  println("Erro no Cholesky: $E_tri - Deveria ser perto de 1e-16")
  println("Erro na resolução do sistema: $E_sis - Deveria ser perto de 1e-16")
  println("Alocações: $all - Deveria ser 0")
end

main()
