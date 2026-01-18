function item=getItem(py_object, index)

get_item = py.getattr(py_object, "__getitem__");
item = get_item(index);