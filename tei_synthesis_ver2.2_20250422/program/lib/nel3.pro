function nel3,arr
  nd=size(arr,/n_dimensions)
  case nd of
     3:   out= n_elements(arr[0,0,*])
     4:   out= n_elements(arr[0,0,*,0])
  endcase
  return,out
end
