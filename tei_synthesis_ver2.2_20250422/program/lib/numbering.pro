function numbering,num,order

  out=''
  for ii=order,1,-1 do begin
     a=fix(num)-fix(num)/10^ii*10^ii
     b=a/10^(ii-1)
     out=out+strmid('0123456789',b,1)   
  endfor
  
  return,out

end
