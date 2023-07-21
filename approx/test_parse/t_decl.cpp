
int main()
{
	int N = 200;
	int *a = new int[10000];
	int b = 0;
	char c = 0;
	//#pragma approx ml(infer) in(a[0:10]) out(b)
	b = a[0] + 2;
	//#pragma approx declare tensor_functor(functor_name: [0:10] = ([::5], [0:10+c], [0:c*10:1]))
  	#pragma approx declare tensor_functor(functor_name: [i, 0:6] = ([i], [i], [i], [i], [i], [i]))
  	#pragma approx declare tensor_functor(fivepoint_stencil: [i, j, 0:5] = ([i,j-1:j+2], [i-1,j], [i+1,5]))
  	#pragma approx declare tensor(first_tnesor: functor_name(a[0:N], a[0:N], a[0:N], a[0:N], a[0:N], a[0:N]))

  	#pragma approx declare tensor_functor(functor_name: [0:c+10] = ([0:c,0:c+5:10]))
  	#pragma approx declare tensor_functor(functor_name: [c:c+10:10] = ([0:2:2]))
	#pragma approx declare tensor_functor(functor_name: [c:j:10] = ([0:2:2]))
}
