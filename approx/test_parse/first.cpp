
int main()
{
	int *a = new int[10000];
	int b = 0;
	char c = 0;
	//#pragma approx ml(infer) in(a[0:10]) out(b)
	b = a[0] + 2;
	#pragma approx declare tensor_functor(functor_name: [0:10] = ([:c:5], [0:10+c], [0:c*10:1]))
	#pragma approx declare tensor_functor(functor_name: [0:10] = ([0:c:5], [0:10+c], [0:c*10:1]))
	#pragma approx declare tensor_functor(functor_name: [0:c+10] = ([0:c]))
	#pragma approx declare tensor_functor(functor_name: [c:c+10:10] = [0::2])
}
