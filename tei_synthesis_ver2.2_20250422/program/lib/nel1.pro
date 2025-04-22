function nel1,arr
  nd=size(arr,/n_dimensions)
  case nd of
     1:   out= n_elements(arr)
     2:   out= n_elements(arr[*,0])
     3:   out= n_elements(arr[*,0,0])
     4:   out= n_elements(arr[*,0,0,0])
  endcase
  return,out
end
