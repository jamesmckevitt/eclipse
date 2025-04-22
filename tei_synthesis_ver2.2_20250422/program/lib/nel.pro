function nel,arr
  nd=size(arr,/n_dimensions)
  case nd of
     0: out=n_elements(arr)
     1: out=n_elements(arr)
  endcase
  return,out
end
