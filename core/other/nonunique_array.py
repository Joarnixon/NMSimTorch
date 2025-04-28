import torch
from copy import copy


class NonuniqueArray:
    """
    A wrapper class for storing elements by reference indices in a PyTorch tensor
    """
    
    def __init__(self, shape, device=None, element_list=None):
        self.data = torch.zeros(shape, dtype=torch.uint8, device=device)
        self.element_list = element_list if element_list is not None else [None]
        self.device = device

    @property
    def shape(self):
        return self.data.shape
    
    def reshape(self, shape):
        self.data = self.data.reshape(shape)
        return self
    
    @classmethod
    def from_tensor(cls, tensor, element_list=None):
        obj = cls(tensor.shape, tensor.device, element_list)
        obj.data = tensor.clone()
        return obj
    
    def clone(self):
        """Create a copy of this NonuniqueArray"""
        new_obj = NonuniqueArray(self.data.shape, self.data.device, self.element_list)
        new_obj.data = self.data.clone()
        return new_obj
    
    def to(self, device):
        """Move to a different device"""
        self.data = self.data.to(device)
        self.device = device
        return self
    
    def __contains__(self, value):
        return value in self.element_list
    
    def __getitem__(self, key):
        sliced_data = self.data[key]
        new_obj = self.__class__(sliced_data.shape, device=self.data.device)
        new_obj.data = sliced_data
        new_obj.element_list = self.element_list
        return new_obj
    
    def __setitem__(self, key, value):
        if isinstance(value, NonuniqueArray):
            # Получаем подмассив целевых позиций
            subarray = self.data[key]
            for element, mask in value.inverse_indices.items():
                if element not in self.element_list:
                    self.element_list.append(element)
                element_index = self.element_list.index(element)

                # Выставляем элемент только в тех позициях, где mask == True
                subarray[mask] = element_index

            self.data[key] = subarray
            return

        if value not in self.element_list:
            self.element_list.append(value)
        element_index = self.element_list.index(value)
        self.data[key] = element_index
        
    def get_value(self, key):
        """Get indices at specified positions"""
        return self.data[key]
    
    def restore(self):
        """Convert indices back to actual elements"""
        result = []
        for i in range(len(self.element_list)):
            mask = self.data == i
            if mask.any():
                result.append((mask, self.element_list[i]))
        return result
    
    def type_matching(self, type_cls):
        """Return mask where elements match the given type"""
        match = torch.zeros_like(self.data, dtype=torch.bool)
        for index, element in enumerate(self.element_list):
            if isinstance(element, type_cls):
                match |= (self.data == index)
        return match
    
    def matching(self, value):
        """Return mask where elements match the given value"""
        if value not in self:
            return torch.zeros_like(self.data, dtype=torch.bool)
        index = self.element_list.index(value)
        return self.data == index
    
    @property
    def inverse_indices(self):
        """Return dictionary of {element: mask} pairs"""
        inverse_dict = {}
        for index, element in enumerate(self.element_list):
            mask = (self.data == index)
            inverse_dict[element] = mask
        return inverse_dict
