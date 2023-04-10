# -*- coding:utf-8 _*-

class Dict(dict):
    def __getattr__(self, key):
        value = self.get(key)
        return Dict(value) if isinstance(value,dict) else value
    def __setattr__(self, key, value):
        self[key] = value

